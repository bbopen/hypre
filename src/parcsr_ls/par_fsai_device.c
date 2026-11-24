/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_fsai.h"
#include <float.h>

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define mat_(l, k, i, j) mat_data[l * (l * k + i) + j]
#define rhs_(l, i, j)    rhs_data[l * i + j]
#define sol_(l, i, j)    sol_data[l * i + j]
#define ls_(i, j)        ls_data[batch_dim * j + i]

#define HYPRE_THRUST_ZIP3(A, B, C) thrust::make_zip_iterator(thrust::make_tuple(A, B, C))

/*--------------------------------------------------------------------------
 * hypre_GPUKernelBatchedGaussJordanSolve
 *--------------------------------------------------------------------------*/

__global__ void
__launch_bounds__(1024, 1)
hypre_GPUKernelBatchedGaussJordanSolve( hypre_DeviceItem  &item,
                                        HYPRE_Int          batch_num_items,
                                        HYPRE_Int          batch_dim,
                                        HYPRE_Complex     *mat_data,
                                        HYPRE_Complex     *rhs_data,
                                        HYPRE_Complex     *sol_data )
{
   HYPRE_UNUSED_VAR(item);

   extern __shared__ void* shmem[];

   HYPRE_Complex    *ls_data = (HYPRE_Complex*) shmem;
   HYPRE_Complex    *coef    = (HYPRE_Complex*) (ls_data + batch_dim * (batch_dim + 1));
   HYPRE_Int        *pos     = (HYPRE_Int*) (coef + 2);

   HYPRE_Int         tidx    = threadIdx.x;
   HYPRE_Int         tidy    = threadIdx.y;
   HYPRE_Int         btid    = blockIdx.y * gridDim.x + blockIdx.x;

   HYPRE_Int         i, k;
   HYPRE_Int         posA;
   HYPRE_Complex     coefA, coefB;
   HYPRE_Complex    *ptrA;

   if (btid < batch_num_items)
   {
      /* Shift to LS belonging to the current batch ID (btid) */
      mat_data += btid * batch_dim * batch_dim;
      rhs_data += btid * batch_dim;
      sol_data += btid * batch_dim;

      /* Copy matrix into shared memory */
      if (tidy < batch_dim)
      {
         ls_(tidx, tidy) = mat_data[tidy * batch_dim + tidx];
      }

      /* Copy RHS into shared memory */
      if (tidy == batch_dim)
      {
         ls_(tidx, tidy) = rhs_data[tidx];
      }

      /* Perform elimination */
      for (k = 0; k < batch_dim; k++)
      {
         /* Pivot computation */
         __syncthreads();
         if ((tidx < 2) && (tidy == 0))
         {
            i = k + 1 + tidx;
            posA  = k;
            ptrA  = &ls_(i, k);
            coefA = fabs(ls_(k, k));

#pragma unroll 1
            for (; i < batch_dim; i += 2)
            {
               coefB = fabs(*ptrA);
               if (coefA < coefB)
               {
                  coefA = coefB;
                  posA  = i;
               }
               ptrA += 2;
            }
            pos[tidx]  = posA;
            coef[tidx] = coefA;
         }

         /* Swap row coefficients */
         __syncthreads();
         if ((tidx == k) && (tidy >= k))
         {
            posA = (coef[1] > coef[0]) ? pos[1] : pos[0];

            coefA = ls_(posA, tidy);
            ls_(posA, tidy) = ls_(tidx, tidy);
            ls_(tidx, tidy) = coefA;
         }

         /* Row scaling */
         __syncthreads();
         if ((tidx == k) && (tidy > k))
         {
            ls_(tidx, tidy) = ls_(tidx, tidy) * (1.0 / ls_(tidx, k));
         }

         /* Row elimination */
         __syncthreads();
         if ((tidx != k) && (tidy > k))
         {
            ls_(tidx, tidy) -= ls_(tidx, k) * ls_(k, tidy);
         }
      }

      __syncthreads();
      if (tidy == batch_dim)
      {
         sol_data[tidx] = ls_(tidx, batch_dim);
      }
   }
}

/*--------------------------------------------------------------------
 * hypre_GPUKernelFSAIExtractSubSystems
 *
 * Output:
 *   1) mat_data: dense matrix coefficients.
 *   2) rhs_data: right hand side coefficients.
 *   3) G_r: number of nonzero coefficients per row of the matrix G.
 *
 * TODO:
 *   1) Minimize intra-warp divergence.
 *--------------------------------------------------------------------*/

__global__ void
hypre_GPUKernelFSAIExtractSubSystems( hypre_DeviceItem &item,
                                      HYPRE_Int         num_rows,
                                      HYPRE_Int        *A_i,
                                      HYPRE_Int        *A_j,
                                      HYPRE_Complex    *A_a,
                                      HYPRE_Int        *P_i,
                                      HYPRE_Int        *P_e,
                                      HYPRE_Int        *P_j,
                                      HYPRE_Int         batch_dim,
                                      HYPRE_Complex    *mat_data,
                                      HYPRE_Complex    *rhs_data,
                                      HYPRE_Int        *G_r )
{
   HYPRE_Int      lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int      i, j, jj, k;
   HYPRE_Int      pj, qj;
   HYPRE_Int      pk, qk;
   HYPRE_Int      A_col, P_col;
   HYPRE_Complex  val;
   hypre_mask     bitmask;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_warp_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      /* Set identity matrix */
      for (j = lane; j < batch_dim; j += HYPRE_WARP_SIZE)
      {
         mat_(batch_dim, i, j, j) = 1.0;
      }

      if (lane == 0)
      {
         pj = read_only_load(P_i + i);
         qj = read_only_load(P_e + i);
      }
      qj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, qj, 0, HYPRE_WARP_SIZE);
      pj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 0, HYPRE_WARP_SIZE);

      if (lane < 2)
      {
         pk = read_only_load(A_i + i + lane);
      }
      qk = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pk, 1, HYPRE_WARP_SIZE);
      pk = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pk, 0, HYPRE_WARP_SIZE);

      /* Set right hand side vector */
      for (j = pj; j < qj; j++)
      {
         if (lane == 0)
         {
            P_col = read_only_load(P_j + j);
         }
         P_col = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, P_col, 0, HYPRE_WARP_SIZE);

         for (k = pk + lane;
              warp_any_sync(item, HYPRE_WARP_FULL_MASK, k < qk);
              k += HYPRE_WARP_SIZE)
         {
            if (k < qk)
            {
               A_col = read_only_load(A_j + k);
            }
            else
            {
               A_col = -1;
            }

            bitmask = hypre_ballot_sync(item, HYPRE_WARP_FULL_MASK, A_col == P_col);
            if (bitmask > 0)
            {
               if (lane == (hypre_ffs(bitmask) - 1))
               {
                  rhs_(batch_dim, i, j - pj) = - read_only_load(A_a + k);
               }
               break;
            }
         }
      }

      /* Loop over requested rows */
      for (j = pj; j < qj; j++)
      {
         if (lane < 2)
         {
            pk = read_only_load(A_i + P_j[j] + lane);
         }
         qk = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pk, 1, HYPRE_WARP_SIZE);
         pk = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pk, 0, HYPRE_WARP_SIZE);

         /* Visit only the lower triangular part */
         for (jj = pj; jj <= j; jj++)
         {
            if (lane == 0)
            {
               P_col = read_only_load(P_j + jj);
            }
            P_col = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, P_col, 0, HYPRE_WARP_SIZE);

            for (k = pk + lane;
                 warp_any_sync(item, HYPRE_WARP_FULL_MASK, k < qk);
                 k += HYPRE_WARP_SIZE)
            {
               if (k < qk)
               {
                  A_col = read_only_load(A_j + k);
               }
               else
               {
                  A_col = -1;
               }

               bitmask = hypre_ballot_sync(item, HYPRE_WARP_FULL_MASK, A_col == P_col);
               if (bitmask > 0)
               {
                  if (lane == (hypre_ffs(bitmask) - 1))
                  {
                     val = read_only_load(A_a + k);
                     mat_(batch_dim, i, j - pj, jj - pj) = val;
                     mat_(batch_dim, i, jj - pj, j - pj) = val;
                  }
                  break;
               }
            }
         }
      }

      /* Set number of nonzero coefficients per row of G */
      if (lane == 0)
      {
         G_r[i] = qj - pj + 1;
      }
   } /* Grid-stride loop over matrix rows */
}

/*--------------------------------------------------------------------
 * hypre_GPUKernelFSAIScaling
 *
 * TODO: unroll inner loop
 *       Use fma?
 *--------------------------------------------------------------------*/

__global__ void
hypre_GPUKernelFSAIScaling( hypre_DeviceItem &item,
                            HYPRE_Int         num_rows,
                            HYPRE_Int         batch_dim,
                            HYPRE_Complex    *sol_data,
                            HYPRE_Complex    *rhs_data,
                            HYPRE_Complex    *scaling,
                            HYPRE_Int        *info )
{
   HYPRE_Int      i, j;
   HYPRE_Complex  val;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      val = scaling[i];
      for (j = 0; j < batch_dim; j++)
      {
         val += sol_(batch_dim, i, j) * rhs_(batch_dim, i, j);
      }

      if (val > 0)
      {
         scaling[i] = 1.0 / sqrt(val);
      }
      else
      {
         scaling[i] = 1.0 / sqrt(scaling[i]);
         info[i] = 1;
      }
   }
}

/*--------------------------------------------------------------------
 * hypre_GPUKernelFSAIGatherEntries
 *
 * Output:
 *   1) G_j: column indices of G_diag
 *   2) G_a: coefficients of G_diag
 *
 * TODO:
 *   1) Use a (sub-)warp per row of G
 *--------------------------------------------------------------------*/

__global__ void
hypre_GPUKernelFSAIGatherEntries( hypre_DeviceItem &item,
                                  HYPRE_Int         num_rows,
                                  HYPRE_Int         batch_dim,
                                  HYPRE_Complex    *sol_data,
                                  HYPRE_Complex    *scaling,
                                  HYPRE_Int        *K_i,
                                  HYPRE_Int        *K_e,
                                  HYPRE_Int        *K_j,
                                  HYPRE_Int        *G_i,
                                  HYPRE_Int        *G_j,
                                  HYPRE_Complex    *G_a )
{
   HYPRE_Int      i, j;
   HYPRE_Int      cnt, il;
   HYPRE_Int      col;
   HYPRE_Complex  val;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      /* Set scaling factor */
      val = scaling[i];

      /* Set diagonal coefficient */
      cnt = G_i[i];
      G_j[cnt] = i;
      G_a[cnt] = val;
      cnt++;

      /* Set off-diagonal coefficients */
      il = 0;
      for (j = K_i[i]; j < K_e[i]; j++)
      {
         col = K_j[j];

         G_j[cnt + il] = col;
         G_a[cnt + il] = sol_(batch_dim, i, il) * val;
         il++;
      }
   }
}

/*--------------------------------------------------------------------
 * hypre_GPUKernelFSAITruncateCandidateOrdered
 *
 * Truncates the candidate pattern matrix (K). This function extracts
 * lower triangular portion of the matrix up to the largest
 * "max_nonzeros_row" coefficients in absolute value.
 *
 * Assumptions:
 *    1) columns are ordered with decreasing absolute coef. values
 *    2) max_nonzeros_row < warp_size.
 *
 * TODO:
 *    1) Perform truncation with COO matrix
 *    2) Use less than one warp per row when possible
 *--------------------------------------------------------------------*/

__global__ void
hypre_GPUKernelFSAITruncateCandidateOrdered( hypre_DeviceItem &item,
                                             HYPRE_Int         max_nonzeros_row,
                                             HYPRE_Int         num_rows,
                                             HYPRE_Int        *K_i,
                                             HYPRE_Int        *K_j,
                                             HYPRE_Complex    *K_a )
{
   HYPRE_Int      lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int      p = 0;
   HYPRE_Int      q = 0;
   HYPRE_Int      i, j, k, kk, cnt;
   HYPRE_Int      col;
   hypre_mask     bitmask;
   HYPRE_Complex  val;
   HYPRE_Int      max_lane;
   HYPRE_Int      max_idx;
   HYPRE_Complex  max_val;
   HYPRE_Complex  warp_max_val;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_warp_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      if (lane < 2)
      {
         p = read_only_load(K_i + i + lane);
      }
      q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1, HYPRE_WARP_SIZE);
      p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0, HYPRE_WARP_SIZE);

      k = 0;
      while (k < max_nonzeros_row)
      {
         /* Initialize variables */
         j = p + k + lane;
         max_val = 0.0;
         max_idx = -1;

         /* Find maximum val/col pair in each lane */
         if (j < q)
         {
            if (K_j[j] < i)
            {
               max_val = abs(K_a[j]);
               max_idx = j;
            }
         }

         for (j += HYPRE_WARP_SIZE; j < q; j += HYPRE_WARP_SIZE)
         {
            if (K_j[j] < i)
            {
               val = abs(K_a[j]);
               if (val > max_val)
               {
                  max_val = val;
                  max_idx = j;
               }
            }
         }

         /* Find maximum coefficient in absolute value in the warp */
         warp_max_val = warp_allreduce_max(item, max_val);

         /* Reorder col/val entries associated with warp_max_val */
         bitmask = hypre_ballot_sync(item, HYPRE_WARP_FULL_MASK, warp_max_val == max_val);
         if (warp_max_val > 0.0)
         {
            cnt = min(hypre_popc(bitmask), max_nonzeros_row - k);

            for (kk = 0; kk < cnt; kk++)
            {
               max_lane = hypre_ffs(bitmask) - 1;
               if (lane == max_lane)
               {
                  col = K_j[p + k + kk];
                  val = K_a[p + k + kk];

                  K_j[p + k + kk] = K_j[max_idx];
                  K_a[p + k + kk] = max_val;

                  K_j[max_idx] = col;
                  K_a[max_idx] = val;
               }

               /* Update bitmask */
               bitmask = hypre_mask_flip_at(bitmask, max_lane);
            }

            /* Update number of nonzeros per row */
            k += cnt;
         }
         else
         {
            break;
         }
      }

      /* Exclude remaining columns */
      for (j = p + k + lane; j < q; j += HYPRE_WARP_SIZE)
      {
         K_j[j] = -1;
      }
   }
}

/*--------------------------------------------------------------------
 * hypre_GPUKernelFSAITruncateCandidateUnordered
 *
 * Truncates the candidate pattern matrix (K). This function extracts
 * lower triangular portion of the matrix up to the largest
 * "max_nonzeros_row" coefficients in absolute value.
 *
 * Assumptions:
 *    1) max_nonzeros_row < warp_size.
 *
 * TODO:
 *    1) Use less than one warp per row when possible
 *--------------------------------------------------------------------*/

__global__ void
hypre_GPUKernelFSAITruncateCandidateUnordered( hypre_DeviceItem &item,
                                               HYPRE_Int         max_nonzeros_row,
                                               HYPRE_Int         num_rows,
                                               HYPRE_Int        *K_i,
                                               HYPRE_Int        *K_e,
                                               HYPRE_Int        *K_j,
                                               HYPRE_Complex    *K_a )
{
   HYPRE_Int      lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int      p = 0;
   HYPRE_Int      q = 0;
   HYPRE_Int      ee, e, i, j, k, kk, cnt;
   hypre_mask     bitmask;
   HYPRE_Complex  val;
   HYPRE_Int      max_lane;
   HYPRE_Int      max_idx;
   HYPRE_Int      max_col;
   HYPRE_Int      colK;
   HYPRE_Complex  valK;
   HYPRE_Complex  max_val;
   HYPRE_Complex  warp_max_val;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_warp_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      if (lane < 2)
      {
         p = read_only_load(K_i + i + lane);
      }
      q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1, HYPRE_WARP_SIZE);
      p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0, HYPRE_WARP_SIZE);

      k = 0;
      while (k < max_nonzeros_row)
      {
         /* Initialize variables */
         j = p + k + lane;
         max_val = 0.0;
         max_idx = -1;

         /* Find maximum val/col pair in each lane */
         if (j < q)
         {
            if (K_j[j] < i)
            {
               max_val = abs(K_a[j]);
               max_idx = j;
            }
         }

         for (j += HYPRE_WARP_SIZE; j < q; j += HYPRE_WARP_SIZE)
         {
            if (K_j[j] < i)
            {
               val = abs(K_a[j]);
               if (val > max_val)
               {
                  max_val = val;
                  max_idx = j;
               }
            }
         }

         /* Find maximum coefficient in absolute value in the warp */
         warp_max_val = warp_allreduce_max(item, max_val);

         /* Reorder col/val entries associated with warp_max_val */
         bitmask = hypre_ballot_sync(item, HYPRE_WARP_FULL_MASK, warp_max_val == max_val);
         if (warp_max_val > 0.0)
         {
            cnt = min(hypre_popc(bitmask), max_nonzeros_row - k);

            for (kk = 0; kk < cnt; kk++)
            {
               max_lane = hypre_ffs(bitmask) - 1;
               if (lane == max_lane)
               {
                  colK = K_j[p + k + kk];
                  valK = K_a[p + k + kk];
                  max_col = K_j[max_idx];

                  if (k + kk == 0)
                  {
                     K_j[p] = max_col;
                     K_a[p] = max_val;

                     K_j[max_idx] = colK;
                     K_a[max_idx] = valK;
                  }
                  else
                  {
                     if (max_col > K_j[p + k + kk - 1])
                     {
                        /* Insert from the right */
                        K_j[p + k + kk] = max_col;
                        K_a[p + k + kk] = max_val;

                        K_j[max_idx] = colK;
                        K_a[max_idx] = valK;
                     }
                     else if (max_col < K_j[p])
                     {
                        /* Insert from the left */
                        for (ee = k + kk; ee > 0; ee--)
                        {
                           K_j[p + ee] = K_j[p + ee - 1];
                           K_a[p + ee] = K_a[p + ee - 1];
                        }

                        K_j[p] = max_col;
                        K_a[p] = max_val;

                        if (max_idx > p + k + kk)
                        {
                           K_j[max_idx] = colK;
                           K_a[max_idx] = valK;
                        }
                     }
                     else
                     {
                        /* Insert in the middle */
                        for (e = k + kk - 1; e >= 0; e--)
                        {
                           if (K_j[p + e] < max_col)
                           {
                              for (ee = k + kk - 1; ee > e; ee--)
                              {
                                 K_j[p + ee + 1] = K_j[p + ee];
                                 K_a[p + ee + 1] = K_a[p + ee];
                              }

                              K_j[p + e + 1] = max_col;
                              K_a[p + e + 1] = max_val;

                              if (max_idx > p + k + kk)
                              {
                                 K_j[max_idx] = colK;
                                 K_a[max_idx] = valK;
                              }

                              break;
                           }
                        }
                     }
                  }
               }

               /* Update bitmask */
               bitmask = hypre_mask_flip_at(bitmask, max_lane);
            }

            /* Update number of nonzeros per row */
            k += cnt;
         }
         else
         {
            break;
         }
      }

      /* Set pointer to the end of this row */
      if (lane == 0)
      {
         K_e[i] = p + k;
      }
   }
}

/*--------------------------------------------------------------------------
 * hypre_BatchedGaussJordanSolveDevice
 *
 * Solve dense linear systems with less than 32 unknowns via Gauss-Jordan
 * elimination.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BatchedGaussJordanSolveDevice( HYPRE_Int       batch_num_items,
                                     HYPRE_Int       batch_dim,
                                     HYPRE_Complex  *mat_data,
                                     HYPRE_Complex  *rhs_data,
                                     HYPRE_Complex  *sol_data )
{
   if (batch_dim > 31)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Error: cannot solve for local systems larger than 31.");
      return hypre_error_flag;
   }

   /* Assign one linear system per thread block*/
   dim3       bDim = hypre_dim3(batch_dim, batch_dim + 1, 1);
   dim3       gDim = hypre_dim3(batch_num_items, 1, 1);
   HYPRE_Int  shared_mem_size = (sizeof(HYPRE_Complex) * ((batch_dim + 1) * batch_dim + 2) +
                                 sizeof(HYPRE_Int) * 2);

   HYPRE_GPU_LAUNCH2(hypre_GPUKernelBatchedGaussJordanSolve, gDim, bDim, shared_mem_size,
                     batch_num_items, batch_dim, mat_data, rhs_data, sol_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIExtractSubSystemsDevice
 *
 * TODO (VPM): This could be a hypre_CSRMatrix routine
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIExtractSubSystemsDevice( HYPRE_Int       num_rows,
                                   HYPRE_Int      *A_i,
                                   HYPRE_Int      *A_j,
                                   HYPRE_Complex  *A_a,
                                   HYPRE_Int      *P_i,
                                   HYPRE_Int      *P_e,
                                   HYPRE_Int      *P_j,
                                   HYPRE_Int       batch_dim,
                                   HYPRE_Complex  *mat_data,
                                   HYPRE_Complex  *rhs_data,
                                   HYPRE_Int      *G_r )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypre_GPUKernelFSAIExtractSubSystems, gDim, bDim, num_rows,
                     A_i, A_j, A_a, P_i, P_e, P_j, batch_dim, mat_data, rhs_data, G_r );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIScalingDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIScalingDevice( HYPRE_Int       num_rows,
                         HYPRE_Int       batch_dim,
                         HYPRE_Complex  *sol_data,
                         HYPRE_Complex  *rhs_data,
                         HYPRE_Complex  *scaling,
                         HYPRE_Int      *info )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypre_GPUKernelFSAIScaling, gDim, bDim,
                     num_rows, batch_dim, sol_data, rhs_data, scaling, info );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIGatherEntriesDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIGatherEntriesDevice( HYPRE_Int       num_rows,
                               HYPRE_Int       batch_dim,
                               HYPRE_Complex  *sol_data,
                               HYPRE_Complex  *scaling,
                               HYPRE_Int      *K_i,
                               HYPRE_Int      *K_e,
                               HYPRE_Int      *K_j,
                               HYPRE_Int      *G_i,
                               HYPRE_Int      *G_j,
                               HYPRE_Complex  *G_a )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypre_GPUKernelFSAIGatherEntries, gDim, bDim,
                     num_rows, batch_dim, sol_data, scaling, K_i, K_e, K_j, G_i, G_j, G_a );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAITruncateCandidateDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAITruncateCandidateDevice( hypre_CSRMatrix *matrix,
                                   HYPRE_Int      **matrix_e,
                                   HYPRE_Int        max_nonzeros_row )
{
   HYPRE_Int      num_rows  = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int     *mat_i     = hypre_CSRMatrixI(matrix);
   HYPRE_Int     *mat_j     = hypre_CSRMatrixJ(matrix);
   HYPRE_Complex *mat_a     = hypre_CSRMatrixData(matrix);

   HYPRE_Int     *mat_e;

   /* Sanity check */
   if (num_rows <= 0)
   {
      *matrix_e = NULL;
      return hypre_error_flag;
   }

   /*-----------------------------------------------------
    * Keep only the largest coefficients in absolute value
    *-----------------------------------------------------*/

   /* Allocate memory for row indices array */
   hypre_GpuProfilingPushRange("Storage1");
   mat_e = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* Mark unwanted entries with -1 */
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "warp", bDim);

   hypre_GpuProfilingPushRange("TruncCand");
   HYPRE_GPU_LAUNCH(hypre_GPUKernelFSAITruncateCandidateUnordered, gDim, bDim,
                    max_nonzeros_row, num_rows, mat_i, mat_e, mat_j, mat_a );
   hypre_GpuProfilingPopRange();

   *matrix_e = mat_e;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupStaticPowerDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupStaticPowerDevice( void               *fsai_vdata,
                                  hypre_ParCSRMatrix *A,
                                  hypre_ParVector    *f,
                                  hypre_ParVector    *u )
{
   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   hypre_ParCSRMatrix     *G                = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag           = hypre_ParCSRMatrixDiag(G);
   HYPRE_Int               local_solve_type = hypre_ParFSAIDataLocalSolveType(fsai_data);
   HYPRE_Int               max_nnz_row      = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   HYPRE_Int               num_levels       = hypre_ParFSAIDataNumLevels(fsai_data);
   HYPRE_Real              threshold        = hypre_ParFSAIDataThreshold(fsai_data);

   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int               num_rows         = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               block_size       = max_nnz_row * max_nnz_row;
   HYPRE_Int               num_nonzeros_G;

   HYPRE_Complex         **sol_aop = NULL;
   HYPRE_Complex         **mat_aop = NULL;

   hypre_ParCSRMatrix     *Atilde;
   hypre_ParCSRMatrix     *B;
   hypre_ParCSRMatrix     *Ktilde;
   hypre_CSRMatrix        *K_diag;
   HYPRE_Int              *K_e = NULL;
   HYPRE_Int               i;

   /* Local linear solve data */
#if defined (HYPRE_USING_MAGMA)
   magma_queue_t          queue     = hypre_HandleMagmaQueue(hypre_handle());
#endif

#if defined (HYPRE_USING_CUSOLVER) || defined (HYPRE_USING_ROCSOLVER)
   vendorSolverHandle_t   vs_handle = hypre_HandleVendorSolverHandle(hypre_handle());
#endif

   /* TODO: Move to fsai_data? */
   HYPRE_Complex          *scaling;
   HYPRE_Int              *info;
   HYPRE_Int              *h_info;

   /* Error code array for FSAI */
   info   = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
   h_info = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------
    *  Sanity checks
    *-----------------------------------------------------*/

   /* Check local linear solve algorithm */
   if (local_solve_type == 1)
   {
#if !(defined (HYPRE_USING_CUSOLVER) || defined(HYPRE_USING_ROCSOLVER))
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "local_solve_type == 1 requires cuSOLVER (CUDA) or rocSOLVER (HIP)\n");
      return hypre_error_flag;
#endif
   }
   else if (local_solve_type == 2)
   {
#if !defined (HYPRE_USING_MAGMA)
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "local_solve_type == 2 requires MAGMA\n");
      return hypre_error_flag;
#endif
   }
   else if (local_solve_type == 0)
   {
      if (max_nnz_row > 31)
      {
         hypre_ParFSAIDataMaxNnzRow(fsai_data) = max_nnz_row = 31;
      }
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown local linear solve type!\n");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------
    *  Compute candidate pattern
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("CandPat");

   /* Compute filtered version of A */
   Atilde = hypre_ParCSRMatrixClone(A, 1);

   /* Pre-filter to reduce SpGEMM cost */
   if (num_levels > 1)
   {
      hypre_ParCSRMatrixDropSmallEntriesDevice(Atilde, threshold, 2);
   }

   /* TODO: Check if Atilde is diagonal */

   /* Compute power pattern */
   switch (num_levels)
   {
      case 1:
         Ktilde = Atilde;
         break;

      case 2:
         Ktilde = hypre_ParCSRMatMatDevice(Atilde, Atilde);
         break;

      case 3:
         /* First pass */
         B = hypre_ParCSRMatMatDevice(Atilde, Atilde);

         /* Second pass */
         Ktilde = hypre_ParCSRMatMatDevice(Atilde, B);
         hypre_ParCSRMatrixDestroy(B);
         break;

      case 4:
         /* First pass */
         B = hypre_ParCSRMatMatDevice(Atilde, Atilde);
         hypre_ParCSRMatrixDropSmallEntriesDevice(B, threshold, 2);

         /* Second pass */
         Ktilde = hypre_ParCSRMatMatDevice(B, B);
         hypre_ParCSRMatrixDestroy(B);
         break;

      default:
         Ktilde = hypre_ParCSRMatrixClone(Atilde, 1);
         for (i = 1; i < num_levels; i++)
         {
            /* Compute temporary matrix */
            B = hypre_ParCSRMatMatDevice(Atilde, Ktilde);

            /* Update resulting matrix */
            hypre_ParCSRMatrixDestroy(Ktilde);
            Ktilde = hypre_ParCSRMatrixClone(B, 1);
         }
   }

   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Filter candidate pattern
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("FilterPat");

#if defined (DEBUG_FSAI)
   {
      hypre_ParCSRMatrixPrintIJ(Ktilde, 0, 0, "FSAI.out.H.ij");
   }
#endif

   /* Set pattern matrix diagonal matrix */
   K_diag = hypre_ParCSRMatrixDiag(Ktilde);

   /* Filter candidate pattern */
   hypre_FSAITruncateCandidateDevice(K_diag, &K_e, max_nnz_row);

#if defined (DEBUG_FSAI)
   {
      hypre_ParCSRMatrixPrintIJ(Ktilde, 0, 0, "FSAI.out.K.ij");
   }
#endif

   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Preprocess input matrix
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("PreProcessA");

   /* TODO: implement faster diagonal extraction (use "i == A_j[A_i[i]]")*/
   scaling = hypre_TAlloc(HYPRE_Complex, num_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixExtractDiagonalDevice(A_diag, scaling, 0);

   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Extract local linear systems
    *-----------------------------------------------------*/

   /* Allocate storage */
   hypre_GpuProfilingPushRange("Storage1");
   HYPRE_Complex  *mat_data = hypre_CTAlloc(HYPRE_Complex,
                                            block_size * num_rows,
                                            HYPRE_MEMORY_DEVICE);
   HYPRE_Complex  *rhs_data = hypre_CTAlloc(HYPRE_Complex, max_nnz_row * num_rows,
                                            HYPRE_MEMORY_DEVICE);
   HYPRE_Complex  *sol_data = hypre_CTAlloc(HYPRE_Complex, max_nnz_row * num_rows,
                                            HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* Gather dense linear subsystems */
   hypre_GpuProfilingPushRange("ExtractLS");
   hypre_FSAIExtractSubSystemsDevice(num_rows,
                                     hypre_CSRMatrixI(A_diag),
                                     hypre_CSRMatrixJ(A_diag),
                                     hypre_CSRMatrixData(A_diag),
                                     hypre_CSRMatrixI(K_diag),
                                     K_e,
                                     hypre_CSRMatrixJ(K_diag),
                                     max_nnz_row,
                                     mat_data,
                                     rhs_data,
                                     hypre_CSRMatrixI(G_diag) + 1);
   hypre_GpuProfilingPopRange();

   /* Copy rhs to solution vector */
   hypre_GpuProfilingPushRange("CopyRHS");
   hypre_TMemcpy(sol_data, rhs_data, HYPRE_Complex, max_nnz_row * num_rows,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* Build array of pointers */
   if (local_solve_type != 0)
   {
      hypre_GpuProfilingPushRange("Storage2");
      sol_aop = hypre_TAlloc(HYPRE_Complex *, num_rows, HYPRE_MEMORY_DEVICE);
      mat_aop = hypre_TAlloc(HYPRE_Complex *, num_rows, HYPRE_MEMORY_DEVICE);
      hypre_GpuProfilingPopRange();

      hypre_GpuProfilingPushRange("FormAOP");
      hypre_ComplexArrayToArrayOfPtrsDevice(num_rows, block_size, mat_data, mat_aop);
      hypre_ComplexArrayToArrayOfPtrsDevice(num_rows, max_nnz_row, sol_data, sol_aop);
      hypre_GpuProfilingPopRange();
   }

   /*-----------------------------------------------------
    *  Solve local linear systems
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("BatchedSolve");
   if (num_rows)
   {
      hypre_GpuProfilingPushRange("Factorization");

      if (local_solve_type == 1)
      {
#if defined (HYPRE_USING_CUSOLVER)
         HYPRE_CUSOLVER_CALL(hypre_cusolver_dnpotrf_batched(vs_handle,
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            max_nnz_row,
                                                            mat_aop,
                                                            max_nnz_row,
                                                            info,
                                                            num_rows));

#elif defined (HYPRE_USING_ROCSOLVER)
         HYPRE_ROCSOLVER_CALL(hypre_rocsolver_potrf_batched(vs_handle,
                                                            rocblas_fill_lower,
                                                            max_nnz_row,
                                                            mat_aop,
                                                            max_nnz_row,
                                                            info,
                                                            num_rows));
#endif
      }
      else if (local_solve_type == 2)
      {
#if defined (HYPRE_USING_MAGMA)
         HYPRE_MAGMA_CALL(magma_dpotrf_batched(MagmaLower,
                                               max_nnz_row,
                                               mat_aop,
                                               max_nnz_row,
                                               info,
                                               num_rows,
                                               queue));
#endif
      }
      hypre_GpuProfilingPopRange(); /* Factorization */

#if defined (HYPRE_DEBUG)
      hypre_TMemcpy(h_info, info, HYPRE_Int, num_rows,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (HYPRE_Int k = 0; k < num_rows; k++)
      {
         if (h_info[k] != 0)
         {
            hypre_printf("Cholesky factorization failed at system #%d, subrow %d\n",
                         k, h_info[k]);
         }
      }
#endif

      hypre_GpuProfilingPushRange("Solve");

      if (local_solve_type == 0)
      {
         hypre_BatchedGaussJordanSolveDevice(num_rows, max_nnz_row, mat_data, rhs_data, sol_data);
      }
      else if (local_solve_type == 1)
      {
#if defined (HYPRE_USING_CUSOLVER)
         HYPRE_CUSOLVER_CALL(hypre_cusolver_dnpotrs_batched(vs_handle,
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            max_nnz_row,
                                                            1,
                                                            mat_aop,
                                                            max_nnz_row,
                                                            sol_aop,
                                                            max_nnz_row,
                                                            info,
                                                            num_rows));
#elif defined (HYPRE_USING_ROCSOLVER)
         HYPRE_ROCSOLVER_CALL(hypre_rocsolver_potrs_batched(vs_handle,
                                                            rocblas_fill_lower,
                                                            max_nnz_row,
                                                            1,
                                                            mat_aop,
                                                            max_nnz_row,
                                                            sol_aop,
                                                            max_nnz_row,
                                                            num_rows));
#endif
      }
      else if (local_solve_type == 2)
      {
#if defined (HYPRE_USING_MAGMA)
         HYPRE_MAGMA_CALL(magma_dpotrs_batched(MagmaLower,
                                               max_nnz_row,
                                               1,
                                               mat_aop,
                                               max_nnz_row,
                                               sol_aop,
                                               max_nnz_row,
                                               num_rows,
                                               queue));
#endif
      }
      hypre_GpuProfilingPopRange(); /* Solve */

#if defined (HYPRE_DEBUG)
      hypre_TMemcpy(h_info, info, HYPRE_Int, num_rows,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (HYPRE_Int k = 0; k < num_rows; k++)
      {
         if (h_info[k] != 0)
         {
            hypre_printf("Cholesky solution failed at system #%d with code %d\n",
                         k, h_info[k]);
         }
      }
#endif
   }
   hypre_GpuProfilingPopRange(); /* BatchedSolve */

   /*-----------------------------------------------------
    *  Finalize construction of the triangular factor
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("BuildFSAI");

   /* Update scaling factor */
   hypre_FSAIScalingDevice(num_rows, max_nnz_row, sol_data, rhs_data, scaling, info);

   /* Compute the row pointer G_i */
   hypre_IntegerInclusiveScanDevice(num_rows + 1, hypre_CSRMatrixI(G_diag));

   /* Get the actual number of nonzero coefficients of G_diag */
   hypre_TMemcpy(&num_nonzeros_G, hypre_CSRMatrixI(G_diag) + num_rows,
                 HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   /* Update the nonzero count of matrix G */
   hypre_CSRMatrixNumNonzeros(G_diag) = num_nonzeros_G;

   /* Set column indices and coefficients of G */
   hypre_FSAIGatherEntriesDevice(num_rows,
                                 max_nnz_row,
                                 sol_data,
                                 scaling,
                                 hypre_CSRMatrixI(K_diag),
                                 K_e,
                                 hypre_CSRMatrixJ(K_diag),
                                 hypre_CSRMatrixI(G_diag),
                                 hypre_CSRMatrixJ(G_diag),
                                 hypre_CSRMatrixData(G_diag));

   hypre_GpuProfilingPopRange();
   /* TODO: Reallocate memory for G_j/G_a? */

   /*-----------------------------------------------------
    *  Free memory
    *-----------------------------------------------------*/

   hypre_ParCSRMatrixDestroy(Ktilde);
   if (num_levels > 1)
   {
      hypre_ParCSRMatrixDestroy(Atilde);
   }

   /* TODO: can we free some of these earlier? */
   hypre_TFree(K_e, HYPRE_MEMORY_DEVICE);
   hypre_TFree(rhs_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(sol_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(mat_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(sol_aop, HYPRE_MEMORY_DEVICE);
   hypre_TFree(mat_aop, HYPRE_MEMORY_DEVICE);
   hypre_TFree(scaling, HYPRE_MEMORY_DEVICE);
   hypre_TFree(info, HYPRE_MEMORY_DEVICE);
   hypre_TFree(h_info, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

#if !defined(HYPRE_COMPLEX)

/*--------------------------------------------------------------------------
 * BitFSAI (algo_type == 4) device kernels (CUDA / HIP)
 *--------------------------------------------------------------------------*/

constexpr HYPRE_Int HYPRE_BITFSAI_ERR_INVALID_DIAG = 1;
constexpr HYPRE_Int HYPRE_BITFSAI_ERR_UNREP        = 2;
constexpr HYPRE_Int HYPRE_BITFSAI_FLAG_WIDE_APPLY  = 4;
constexpr HYPRE_Int HYPRE_BITFSAI_ERR_NONSYMMETRIC = 8;

__device__ __forceinline__ bool
hypre_BitFSAIIsFiniteDevice( double v )
{
   return (v == v && v >= -DBL_MAX && v <= DBL_MAX);
}

__device__ __forceinline__ bool
hypre_BitFSAIIsStorableF32Device( double v )
{
   float s = (float) v;
   return (hypre_BitFSAIIsFiniteDevice(v) && s == s && s != 0.0f && s >= -FLT_MAX && s <= FLT_MAX);
}

__device__ __forceinline__ HYPRE_Int
hypre_BitFSAIProjectRowDevice( HYPRE_Int     m,
                               const double *q_dec,
                               const double *L,
                               const double *A_patch,
                               double       *diag_out,
                               double       *scale_out )
{
   constexpr HYPRE_Int MAX_PATCH = 16;
   if (m < 2)
   {
      return 0;
   }
   double c22 = A_patch[(m - 1) * MAX_PATCH + (m - 1)];
   if (!(c22 > 0.0) || !hypre_BitFSAIIsFiniteDevice(c22))
   {
      return -1;
   }

   double c11 = 0.0;
   double c12 = 0.0;
   for (HYPRE_Int b = 0; b < m - 1; b++)
   {
      double ub = 0.0;
      for (HYPRE_Int a = b; a < m - 1; a++)
      {
         ub += L[a * MAX_PATCH + b] * q_dec[a];
      }
      c11 += ub * ub;
      c12 += ub * L[(m - 1) * MAX_PATCH + b];
   }
   if (!(c11 > 0.0) || !hypre_BitFSAIIsFiniteDevice(c11) || !hypre_BitFSAIIsFiniteDevice(c12))
   {
      return 0;
   }

   double projection = c12 / c11;
   if (!hypre_BitFSAIIsFiniteDevice(projection))
   {
      return 0;
   }

   double residual_sq = 0.0;
   for (HYPRE_Int b = 0; b < m; b++)
   {
      double ub = 0.0;
      if (b < m - 1)
      {
         for (HYPRE_Int a = b; a < m - 1; a++)
         {
            ub += L[a * MAX_PATCH + b] * q_dec[a];
         }
      }
      double res = L[(m - 1) * MAX_PATCH + b] - projection * ub;
      residual_sq += res * res;
   }
   if (!(residual_sq > 0.0) || !hypre_BitFSAIIsFiniteDevice(residual_sq))
   {
      return 0;
   }

   double g_opt = 1.0 / sqrt(residual_sq);
   double s_opt = -projection * g_opt;
   if (!(g_opt > 0.0) || !hypre_BitFSAIIsFiniteDevice(g_opt) || !hypre_BitFSAIIsFiniteDevice(s_opt))
   {
      return 0;
   }

   float g_f32 = (float) g_opt;
   float s_f32 = (float) s_opt;
   if (s_f32 == 0.0f)
   {
      return 0;
   }
   if (!hypre_BitFSAIIsStorableF32Device(g_opt) || !hypre_BitFSAIIsStorableF32Device(s_opt))
   {
      return -2;
   }

   double sg = (double) g_f32;
   double ss = (double) s_f32;
   double energy = 0.0;
   double comp = 0.0;
   double sum_abs = 0.0;
   for (HYPRE_Int a = 0; a < m; a++)
   {
      double wa = (a == m - 1) ? sg : (ss * q_dec[a]);
      for (HYPRE_Int b = 0; b < m; b++)
      {
         double wb = (b == m - 1) ? sg : (ss * q_dec[b]);
         double a_ab = (a >= b) ? A_patch[a * MAX_PATCH + b] : A_patch[b * MAX_PATCH + a];
         double term = wa * a_ab * wb;
         if (!hypre_BitFSAIIsFiniteDevice(term))
         {
            return -2;
         }
         double adj = term - comp;
         double next = energy + adj;
         comp = (next - energy) - adj;
         energy = next;
         sum_abs += fabs(term);
      }
   }
   if (!hypre_BitFSAIIsFiniteDevice(energy) || !hypre_BitFSAIIsFiniteDevice(sum_abs))
   {
      return -2;
   }
   double bound = 8.0 * DBL_EPSILON * sum_abs;
   if (!(fabs(energy - 1.0) + bound <= 1.0e-4))
   {
      return 0;
   }

   *diag_out  = g_opt;
   *scale_out = s_opt;
   return 1;
}

__device__ __forceinline__ HYPRE_Int
hypre_BitFSAIFindPosDevice( const HYPRE_Int *A_i,
                            const HYPRE_Int *A_j,
                            HYPRE_Int        row,
                            HYPRE_Int        target_col )
{
   HYPRE_Int rs = read_only_load(A_i + row);
   HYPRE_Int re = read_only_load(A_i + row + 1);
   for (HYPRE_Int k = rs; k < re; k++)
   {
      if (read_only_load(A_j + k) == target_col)
      {
         return k;
      }
   }
   return -1;
}

__global__ void
hypre_GPUKernelBitFSAISetup( hypre_DeviceItem    &item,
                             HYPRE_Int            num_rows,
                             HYPRE_Int            patch_cap,
                             const HYPRE_Int     *A_i,
                             const HYPRE_Int     *A_j,
                             const HYPRE_Complex *A_a,
                             float               *g_diag,
                             float               *g_scale,
                             unsigned char       *q_tmp,
                             HYPRE_Int           *err_flags )
{
   constexpr HYPRE_Int MAX_PATCH = 16;

   for (HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      HYPRE_Int     J[MAX_PATCH];
      HYPRE_Int     Klow[MAX_PATCH];
      HYPRE_Int     Kup[MAX_PATCH];
      double        a_off[MAX_PATCH];
      double        A_patch[MAX_PATCH * MAX_PATCH];
      double        L[MAX_PATCH * MAX_PATCH];
      double        g_off[MAX_PATCH];
      double        q_dec[MAX_PATCH];
      unsigned char q_code[MAX_PATCH];

      HYPRE_Int rs    = read_only_load(A_i + i);
      HYPRE_Int re    = read_only_load(A_i + i + 1);
      HYPRE_Int m_off = 0;
      double    a_ii  = 0.0;
      bool      found_diag = false;
      bool      diagonal_only = false;
      double    g_d   = 0.0;
      double    scale = 0.0;

      for (HYPRE_Int k = rs; k < re; k++)
      {
         HYPRE_Int col = read_only_load(A_j + k);
         double    val = (double) read_only_load(A_a + k);
         if (col == i)
         {
            a_ii = val;
            found_diag = true;
         }
         else if (col < i)
         {
            HYPRE_Int k_up = hypre_BitFSAIFindPosDevice(A_i, A_j, col, i);
            if (k_up < 0)
            {
               atomicOr(err_flags, HYPRE_BITFSAI_ERR_NONSYMMETRIC);
               continue;
            }
            if (m_off < patch_cap - 1)
            {
               J[m_off]     = col;
               Klow[m_off]  = k;
               Kup[m_off]   = k_up;
               a_off[m_off] = val;
               m_off++;
            }
            else if (patch_cap > 1)
            {
               HYPRE_Int min_idx = 0;
               double    min_abs = fabs(a_off[0]);
               for (HYPRE_Int a = 1; a < m_off; a++)
               {
                  double av = fabs(a_off[a]);
                  if (av < min_abs)
                  {
                     min_abs = av;
                     min_idx = a;
                  }
               }
               if (fabs(val) > min_abs)
               {
                  J[min_idx]     = col;
                  Klow[min_idx]  = k;
                  Kup[min_idx]   = k_up;
                  a_off[min_idx] = val;
               }
            }
         }
         else
         {
            if (hypre_BitFSAIFindPosDevice(A_i, A_j, col, i) < 0)
            {
               atomicOr(err_flags, HYPRE_BITFSAI_ERR_NONSYMMETRIC);
            }
         }
      }

      if (!found_diag || !(a_ii > 0.0) || !hypre_BitFSAIIsFiniteDevice(a_ii))
      {
         atomicOr(err_flags, HYPRE_BITFSAI_ERR_INVALID_DIAG);
         g_diag[i]  = 1.0f;
         g_scale[i] = 1.0f;
         continue;
      }

      if (m_off == 0)
      {
         diagonal_only = true;
         g_d   = 1.0 / sqrt(a_ii);
         scale = g_d;
      }
      else
      {
         HYPRE_Int m = m_off + 1;
         for (HYPRE_Int a = 0; a < m; a++)
         {
            for (HYPRE_Int b = 0; b <= a; b++)
            {
               A_patch[a * MAX_PATCH + b] = 0.0;
            }
         }

         for (HYPRE_Int a = 0; a < m_off; a++)
         {
            HYPRE_Int ra  = J[a];
            HYPRE_Int rrs = read_only_load(A_i + ra);
            HYPRE_Int rre = read_only_load(A_i + ra + 1);
            for (HYPRE_Int k = rrs; k < rre; k++)
            {
               HYPRE_Int col = read_only_load(A_j + k);
               for (HYPRE_Int b = 0; b <= a; b++)
               {
                  if (J[b] == col)
                  {
                     A_patch[a * MAX_PATCH + b] = (double) read_only_load(A_a + k);
                     break;
                  }
               }
            }
            A_patch[(m - 1) * MAX_PATCH + a] = a_off[a];
         }
         A_patch[(m - 1) * MAX_PATCH + (m - 1)] = a_ii;

         for (HYPRE_Int a = 0; a < m; a++)
         {
            for (HYPRE_Int b = 0; b <= a; b++)
            {
               L[a * MAX_PATCH + b] = A_patch[a * MAX_PATCH + b];
            }
         }

         bool ok = true;
         for (HYPRE_Int a = 0; a < m && ok; a++)
         {
            for (HYPRE_Int b = 0; b <= a; b++)
            {
               double s = L[a * MAX_PATCH + b];
               for (HYPRE_Int c = 0; c < b; c++)
               {
                  s -= L[a * MAX_PATCH + c] * L[b * MAX_PATCH + c];
               }
               if (a == b)
               {
                  if (!(s > 0.0) || !hypre_BitFSAIIsFiniteDevice(s))
                  {
                     ok = false;
                     break;
                  }
                  L[a * MAX_PATCH + a] = sqrt(s);
               }
               else
               {
                  double lv = s / L[b * MAX_PATCH + b];
                  if (!hypre_BitFSAIIsFiniteDevice(lv))
                  {
                     ok = false;
                     break;
                  }
                  L[a * MAX_PATCH + b] = lv;
               }
            }
         }

         if (ok)
         {
            g_d = 1.0 / L[(m - 1) * MAX_PATCH + (m - 1)];
            if (!hypre_BitFSAIIsFiniteDevice(g_d))
            {
               ok = false;
            }
            else
            {
               for (HYPRE_Int a = m_off - 1; a >= 0; a--)
               {
                  double s = -L[(m - 1) * MAX_PATCH + a] * g_d;
                  for (HYPRE_Int b = a + 1; b < m_off; b++)
                  {
                     s -= L[b * MAX_PATCH + a] * g_off[b];
                  }
                  g_off[a] = s / L[a * MAX_PATCH + a];
                  if (!hypre_BitFSAIIsFiniteDevice(g_off[a]))
                  {
                     ok = false;
                     break;
                  }
               }
            }
         }

         if (!ok)
         {
            diagonal_only = true;
            g_d   = 1.0 / sqrt(a_ii);
            scale = g_d;
         }
         else
         {
            scale = 0.0;
            for (HYPRE_Int a = 0; a < m_off; a++)
            {
               scale = fmax(scale, fabs(g_off[a]));
            }

            if (!(scale > 0.0) || !hypre_BitFSAIIsFiniteDevice(scale))
            {
               diagonal_only = true;
               g_d   = 1.0 / sqrt(a_ii);
               scale = g_d;
            }
            else
            {
               for (HYPRE_Int a = 0; a < m_off; a++)
               {
                  double    u = 7.0 * (g_off[a] / scale);
                  HYPRE_Int q = (HYPRE_Int)(u >= 0.0 ? floor(u + 0.5) : ceil(u - 0.5));
                  if (q > 7)  { q = 7; }
                  if (q < -7) { q = -7; }
                  q_code[a] = (unsigned char)(q + 8);
                  q_dec[a]  = (double)(((float) q) * (1.0f / 7.0f));
               }

               HYPRE_Int proj_rc = hypre_BitFSAIProjectRowDevice(m, q_dec, L, A_patch, &g_d, &scale);
               if (proj_rc == -1)
               {
                  atomicOr(err_flags, HYPRE_BITFSAI_ERR_INVALID_DIAG);
                  g_diag[i]  = 1.0f;
                  g_scale[i] = 1.0f;
                  continue;
               }
               if (proj_rc == -2)
               {
                  atomicOr(err_flags, HYPRE_BITFSAI_ERR_UNREP);
                  g_diag[i]  = 1.0f;
                  g_scale[i] = 1.0f;
                  continue;
               }
               if (proj_rc == 0)
               {
                  diagonal_only = true;
                  g_d   = 1.0 / sqrt(a_ii);
                  scale = g_d;
                  for (HYPRE_Int a = 0; a < m_off; a++)
                  {
                     q_code[a] = 8;
                  }
               }
               else
               {
                  for (HYPRE_Int a = 0; a < m_off; a++)
                  {
                     q_tmp[Klow[a]] = q_code[a];
                     q_tmp[Kup[a]]  = q_code[a];
                  }
               }
            }
         }
      }

      if (!hypre_BitFSAIIsStorableF32Device(g_d) || !hypre_BitFSAIIsStorableF32Device(scale))
      {
         atomicOr(err_flags, HYPRE_BITFSAI_ERR_UNREP);
         g_diag[i]  = 1.0f;
         g_scale[i] = 1.0f;
         continue;
      }
      if (diagonal_only)
      {
         double sd = (double)(float) g_d;
         double se = sd * sd * a_ii;
         if (!hypre_BitFSAIIsFiniteDevice(se) || fabs(se - 1.0) > 1.0e-4)
         {
            atomicOr(err_flags, HYPRE_BITFSAI_ERR_UNREP);
            g_diag[i]  = 1.0f;
            g_scale[i] = 1.0f;
            continue;
         }
      }

      double ratio = fabs(scale) / g_d;
      if (ratio < 1e-6 || ratio > 1e6 ||
          fabs(scale) < sqrt((double) FLT_MIN) ||
          fabs(scale) > sqrt((double) FLT_MAX))
      {
         atomicOr(err_flags, HYPRE_BITFSAI_FLAG_WIDE_APPLY);
      }

      g_diag[i]  = (float) g_d;
      g_scale[i] = (float) scale;
   }
}

__global__ void
hypre_GPUKernelBitFSAIPack( hypre_DeviceItem    &item,
                            HYPRE_Int            nnz,
                            const unsigned char *q_tmp,
                            unsigned char       *q_packed )
{
   HYPRE_Int packed_bytes = (nnz / 2) + (nnz % 2);
   for (HYPRE_Int idx = hypre_gpu_get_grid_thread_id<1, 1>(item);
        idx < packed_bytes;
        idx += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      HYPRE_Int     k0 = idx << 1;
      unsigned char lo = q_tmp[k0] & 0x0F;
      unsigned char hi = (k0 + 1 < nnz) ? (q_tmp[k0 + 1] & 0x0F) : 8;
      q_packed[idx] = (unsigned char)(lo | (hi << 4));
   }
}

__global__ void
hypre_GPUKernelBitFSAIFwd( hypre_DeviceItem    &item,
                           HYPRE_Int            num_rows,
                           const HYPRE_Int     *A_i,
                           const HYPRE_Int     *A_j,
                           const float         *g_diag,
                           const float         *g_scale,
                           const unsigned char *q_packed,
                           const HYPRE_Complex *b_data,
                           float               *y_work )
{
   for (HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      HYPRE_Int rs  = read_only_load(A_i + i);
      HYPRE_Int re  = read_only_load(A_i + i + 1);
      float     sc  = g_scale[i];
      float     acc = 0.0f;

      for (HYPRE_Int k = rs; k < re; k++)
      {
         HYPRE_Int col = read_only_load(A_j + k);
         if (col < i)
         {
            unsigned char byte = read_only_load(q_packed + (k >> 1));
            unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            float         w    = ((float)((int) nib - 8)) * (1.0f / 7.0f);
            acc = fmaf(w, (float) read_only_load(b_data + col), acc);
         }
      }

      float ti  = fmaf(sc, acc, g_diag[i] * (float) b_data[i]);
      y_work[i] = sc * ti;
   }
}

__global__ void
hypre_GPUKernelBitFSAITr( hypre_DeviceItem    &item,
                          HYPRE_Int            num_rows,
                          HYPRE_Complex        alpha,
                          HYPRE_Real           omega,
                          const HYPRE_Int     *A_i,
                          const HYPRE_Int     *A_j,
                          const float         *g_diag,
                          const float         *g_scale,
                          const unsigned char *q_packed,
                          const float         *y_work,
                          HYPRE_Complex       *x_data )
{
   for (HYPRE_Int j = hypre_gpu_get_grid_thread_id<1, 1>(item);
        j < num_rows;
        j += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      HYPRE_Int rs   = read_only_load(A_i + j);
      HYPRE_Int re   = read_only_load(A_i + j + 1);
      float     sc_j = g_scale[j];
      float     acc  = (sc_j != 0.0f) ? (g_diag[j] * (y_work[j] / sc_j)) : 0.0f;

      for (HYPRE_Int k = rs; k < re; k++)
      {
         HYPRE_Int col = read_only_load(A_j + k);
         if (col > j)
         {
            unsigned char byte = read_only_load(q_packed + (k >> 1));
            unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            float         w    = ((float)((int) nib - 8)) * (1.0f / 7.0f);
            acc = fmaf(w, read_only_load(y_work + col), acc);
         }
      }

      if (alpha == (HYPRE_Complex) 0.0)
      {
         x_data[j] = (HYPRE_Complex)((double) omega * acc);
      }
      else
      {
         x_data[j] = (HYPRE_Complex)((double) alpha * (double) x_data[j] +
                                     (double) omega * acc);
      }
   }
}

__global__ void
hypre_GPUKernelBitFSAIFwdWide( hypre_DeviceItem    &item,
                               HYPRE_Int            num_rows,
                               const HYPRE_Int     *A_i,
                               const HYPRE_Int     *A_j,
                               const float         *g_diag,
                               const float         *g_scale,
                               const unsigned char *q_packed,
                               const HYPRE_Complex *b_data,
                               double              *t_work )
{
   for (HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      HYPRE_Int rs  = read_only_load(A_i + i);
      HYPRE_Int re  = read_only_load(A_i + i + 1);
      double    sc  = (double) g_scale[i];
      double    acc = (double) g_diag[i] * (double) b_data[i];

      for (HYPRE_Int k = rs; k < re; k++)
      {
         HYPRE_Int col = read_only_load(A_j + k);
         if (col < i)
         {
            unsigned char byte = read_only_load(q_packed + (k >> 1));
            unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            double        w    = (double)(((float)((int) nib - 8)) * (1.0f / 7.0f));
            acc = fma(sc * w, (double) read_only_load(b_data + col), acc);
         }
      }

      t_work[i] = acc;
   }
}

__global__ void
hypre_GPUKernelBitFSAITrWide( hypre_DeviceItem    &item,
                              HYPRE_Int            num_rows,
                              HYPRE_Complex        alpha,
                              HYPRE_Real           omega,
                              const HYPRE_Int     *A_i,
                              const HYPRE_Int     *A_j,
                              const float         *g_diag,
                              const float         *g_scale,
                              const unsigned char *q_packed,
                              const double        *t_work,
                              HYPRE_Complex       *x_data )
{
   for (HYPRE_Int j = hypre_gpu_get_grid_thread_id<1, 1>(item);
        j < num_rows;
        j += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      HYPRE_Int rs  = read_only_load(A_i + j);
      HYPRE_Int re  = read_only_load(A_i + j + 1);
      double    acc = (double) g_diag[j] * t_work[j];

      for (HYPRE_Int k = rs; k < re; k++)
      {
         HYPRE_Int col = read_only_load(A_j + k);
         if (col > j)
         {
            unsigned char byte = read_only_load(q_packed + (k >> 1));
            unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            double        w    = (double)(((float)((int) nib - 8)) * (1.0f / 7.0f));
            acc = fma((double) read_only_load(g_scale + col) * w, read_only_load(t_work + col), acc);
         }
      }

      if (alpha == (HYPRE_Complex) 0.0)
      {
         x_data[j] = (HYPRE_Complex)((double) omega * acc);
      }
      else
      {
         x_data[j] = (HYPRE_Complex)((double) alpha * (double) x_data[j] +
                                     (double) omega * acc);
      }
   }
}

#endif /* !defined(HYPRE_COMPLEX) */

#endif /* if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */
#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_FSAISetupBitFSAIDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupBitFSAIDevice( void               *fsai_vdata,
                              hypre_ParCSRMatrix *A )
{
#if (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)) && !defined(HYPRE_COMPLEX)
   hypre_ParFSAIData    *fsai_data   = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int             max_nnz_row = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   HYPRE_MemoryLocation  memloc      = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_CSRMatrix      *A_diag      = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             num_rows    = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int             nnz         = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int             patch_cap;
   HYPRE_Int             packed_bytes;
   HYPRE_Int             h_err       = 0;
   HYPRE_Int             wide_apply  = 0;
   HYPRE_Int            *d_err_flags;
   float                *g_diag;
   float                *g_scale;
   unsigned char        *q_tmp;
   unsigned char        *q_packed;
   float                *y_work      = NULL;
   double               *t_work      = NULL;

   hypre_FSAIFreeBitFSAI(fsai_vdata);

   hypre_ParFSAIDataBitFSAIADiag(fsai_data)   = A_diag;
   hypre_ParFSAIDataBitFSAINumRows(fsai_data) = num_rows;
   hypre_ParFSAIDataBitFSAINnz(fsai_data)     = nnz;
   hypre_ParFSAIDataBitFSAIMemLoc(fsai_data)  = memloc;

   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   if (max_nnz_row > 0)
   {
      patch_cap = (max_nnz_row > 16) ? 16 : max_nnz_row;
   }
   else
   {
      double avg_deg = (double) nnz / (double) num_rows;
      patch_cap = (avg_deg > 22.0) ? 8 : 12;
   }

   packed_bytes = (nnz / 2) + (nnz % 2);
   g_diag      = hypre_TAlloc(float, num_rows, memloc);
   g_scale     = hypre_TAlloc(float, num_rows, memloc);
   q_packed    = hypre_TAlloc(unsigned char, (packed_bytes > 0) ? packed_bytes : 1, memloc);
   q_tmp       = hypre_TAlloc(unsigned char, (nnz > 0) ? nnz : 1, memloc);
   d_err_flags = hypre_CTAlloc(HYPRE_Int, 1, memloc);

   if (nnz > 0)
   {
      hypre_Memset(q_tmp, 8, (size_t) nnz * sizeof(unsigned char), memloc);
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   hypre_GpuProfilingPushRange("BitFSAISetup");
   HYPRE_GPU_LAUNCH( hypre_GPUKernelBitFSAISetup, gDim, bDim,
                     num_rows, patch_cap,
                     hypre_CSRMatrixI(A_diag),
                     hypre_CSRMatrixJ(A_diag),
                     hypre_CSRMatrixData(A_diag),
                     g_diag, g_scale, q_tmp, d_err_flags );

   if (packed_bytes > 0)
   {
      dim3 gDimPack = hypre_GetDefaultDeviceGridDimension(packed_bytes, "thread", bDim);
      HYPRE_GPU_LAUNCH( hypre_GPUKernelBitFSAIPack, gDimPack, bDim,
                        nnz, q_tmp, q_packed );
   }
   hypre_GpuProfilingPopRange();

   hypre_TMemcpy(&h_err, d_err_flags, HYPRE_Int, 1, HYPRE_MEMORY_HOST, memloc);
   hypre_TFree(d_err_flags, memloc);
   hypre_TFree(q_tmp, memloc);

   if (h_err & (HYPRE_BITFSAI_ERR_INVALID_DIAG | HYPRE_BITFSAI_ERR_UNREP | HYPRE_BITFSAI_ERR_NONSYMMETRIC))
   {
      hypre_TFree(g_diag, memloc);
      hypre_TFree(g_scale, memloc);
      hypre_TFree(q_packed, memloc);
      hypre_ParFSAIDataBitFSAIADiag(fsai_data)   = NULL;
      hypre_ParFSAIDataBitFSAINumRows(fsai_data) = 0;
      hypre_ParFSAIDataBitFSAINnz(fsai_data)     = 0;
      if (h_err & HYPRE_BITFSAI_ERR_NONSYMMETRIC)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "BitFSAI (algo_type = 4) requires structurally symmetric (mirrored) A_diag\n");
      }
      else if (h_err & HYPRE_BITFSAI_ERR_INVALID_DIAG)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "BitFSAI (algo_type = 4) requires finite strictly positive diagonals\n");
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "BitFSAI (algo_type = 4) factor cannot be represented in FP32\n");
      }
      return hypre_error_flag;
   }

#if !defined(HYPRE_SINGLE)
   wide_apply = 1;
#else
   wide_apply = (h_err & HYPRE_BITFSAI_FLAG_WIDE_APPLY) ? 1 : 0;
#endif

   if (wide_apply)
   {
      t_work = hypre_TAlloc(double, num_rows, memloc);
   }
   else
   {
      y_work = hypre_TAlloc(float, num_rows, memloc);
   }

   hypre_ParFSAIDataBitFSAIWideApply(fsai_data) = wide_apply;
   hypre_ParFSAIDataBitFSAIGDiag(fsai_data)     = g_diag;
   hypre_ParFSAIDataBitFSAIGScale(fsai_data)    = g_scale;
   hypre_ParFSAIDataBitFSAIQPacked(fsai_data)   = q_packed;
   hypre_ParFSAIDataBitFSAIYWork(fsai_data)     = y_work;
   hypre_ParFSAIDataBitFSAITWork(fsai_data)     = t_work;

   return hypre_error_flag;
#else
   HYPRE_UNUSED_VAR(fsai_vdata);
   HYPRE_UNUSED_VAR(A);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "Device BitFSAI (algo_type = 4) requires CUDA or HIP with real arithmetic!\n");
   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * hypre_FSAIApplyBitFSAIDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIApplyBitFSAIDevice( void            *fsai_vdata,
                              HYPRE_Complex    alpha,
                              hypre_ParVector *b,
                              hypre_ParVector *x )
{
#if (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)) && !defined(HYPRE_COMPLEX)
   hypre_ParFSAIData   *fsai_data  = (hypre_ParFSAIData*) fsai_vdata;
   hypre_CSRMatrix     *A_diag     = hypre_ParFSAIDataBitFSAIADiag(fsai_data);
   HYPRE_Int            num_rows   = hypre_ParFSAIDataBitFSAINumRows(fsai_data);
   HYPRE_Int            wide_apply = hypre_ParFSAIDataBitFSAIWideApply(fsai_data);
   const float         *g_diag     = hypre_ParFSAIDataBitFSAIGDiag(fsai_data);
   const float         *g_scale    = hypre_ParFSAIDataBitFSAIGScale(fsai_data);
   const unsigned char *q_packed   = hypre_ParFSAIDataBitFSAIQPacked(fsai_data);
   float               *y_work     = hypre_ParFSAIDataBitFSAIYWork(fsai_data);
   double              *t_work     = hypre_ParFSAIDataBitFSAITWork(fsai_data);
   HYPRE_Real           omega      = hypre_ParFSAIDataOmega(fsai_data);

   if (num_rows <= 0 || !A_diag)
   {
      return hypre_error_flag;
   }

   const HYPRE_Int     *A_i    = hypre_CSRMatrixI(A_diag);
   const HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A_diag);
   const HYPRE_Complex *b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
   HYPRE_Complex       *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

   if (wide_apply)
   {
      HYPRE_GPU_LAUNCH( hypre_GPUKernelBitFSAIFwdWide, gDim, bDim,
                        num_rows, A_i, A_j, g_diag, g_scale, q_packed, b_data, t_work );

      HYPRE_GPU_LAUNCH( hypre_GPUKernelBitFSAITrWide, gDim, bDim,
                        num_rows, alpha, omega, A_i, A_j, g_diag, g_scale, q_packed, t_work, x_data );
   }
   else
   {
      HYPRE_GPU_LAUNCH( hypre_GPUKernelBitFSAIFwd, gDim, bDim,
                        num_rows, A_i, A_j, g_diag, g_scale, q_packed, b_data, y_work );

      HYPRE_GPU_LAUNCH( hypre_GPUKernelBitFSAITr, gDim, bDim,
                        num_rows, alpha, omega, A_i, A_j, g_diag, g_scale, q_packed, y_work, x_data );
   }

   return hypre_error_flag;
#else
   HYPRE_UNUSED_VAR(fsai_vdata);
   HYPRE_UNUSED_VAR(alpha);
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "Device BitFSAI (algo_type = 4) requires CUDA or HIP with real arithmetic!\n");
   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupDevice( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   hypre_ParFSAIData       *fsai_data     = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int                algo_type     = hypre_ParFSAIDataAlgoType(fsai_data);
   hypre_ParCSRMatrix      *G             = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix      *h_A;

   hypre_GpuProfilingPushRange("FSAISetup");

   if (algo_type == 1 || algo_type == 2)
   {
      /* Initialize matrix G on host */
      hypre_ParCSRMatrixInitialize_v2(G, HYPRE_MEMORY_HOST);

      /* Clone input matrix on host */
      h_A = hypre_ParCSRMatrixClone_v2(A, 1, HYPRE_MEMORY_HOST);

      /* Compute FSAI factor on host */
      switch (algo_type)
      {
         case 2:
            hypre_FSAISetupOMPDyn(fsai_vdata, h_A, f, u);
            break;

         default:
            hypre_FSAISetupNative(fsai_vdata, h_A, f, u);
            break;
      }

      /* Move FSAI factor G to device */
      hypre_ParCSRMatrixMigrate(G, HYPRE_MEMORY_DEVICE);

      /* Destroy temporary data on host */
      HYPRE_ParCSRMatrixDestroy(h_A);
   }
   else
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      /* Initialize matrix G on device */
      hypre_ParCSRMatrixInitialize_v2(G, HYPRE_MEMORY_DEVICE);

      if (algo_type == 3)
      {
         hypre_FSAISetupStaticPowerDevice(fsai_vdata, A, f, u);
      }
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Device FSAI not implemented for SYCL!\n");
#endif
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */
