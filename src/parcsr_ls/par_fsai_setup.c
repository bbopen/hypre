/******************************************************************************
 *  Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 *  HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"
#include <float.h>

/*****************************************************************************
 *
 * Routine for driving the setup phase of FSAI
 *
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDenseMat
 *
 * Extract A[P, P] into a dense matrix.
 *
 * Parameters:
 * - A:       The hypre_CSRMatrix whose submatrix will be extracted.
 * - A_sub:   A patt_size^2 - sized array to hold the lower triangular of
 *            the symmetric submatrix A[P, P].
 * - pattern: A patt_size - sized array to hold the wanted rows/cols.
 * - marker:  A work array of length equal to the number of columns in A.
 *            All values should be -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixExtractDenseMat( hypre_CSRMatrix *A,
                                hypre_Vector    *A_sub,
                                HYPRE_Int       *pattern,
                                HYPRE_Int        patt_size,
                                HYPRE_Int       *marker )
{
   HYPRE_Int     *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_a = hypre_CSRMatrixData(A);
   HYPRE_Complex *A_sub_data = hypre_VectorData(A_sub);

   /* Local variables */
   HYPRE_Int      cc, i, ii, j;

   // TODO: Do we need to reinitialize all entries?
   for (i = 0; i < hypre_VectorSize(A_sub); i++)
   {
      A_sub_data[i] = 0.0;
   }

   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
      {
         if ((A_j[j] <= ii) &&
             (cc = marker[A_j[j]]) >= 0)
         {
            A_sub_data[cc * patt_size + i] = A_a[j];
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDenseRow
 *
 * Extract the dense subrow from a matrix (A[i, P])
 *
 * Parameters:
 * - A:         The hypre_CSRMatrix whose subrow will be extracted.
 * - A_subrow:  The extracted subrow of A[i, P].
 * - marker:    A work array of length equal to the number of row in A.
 *              Assumed to be set to all -1.
 * - row_num:   which row index of A we want to extract data from.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixExtractDenseRow( hypre_CSRMatrix *A,
                                hypre_Vector    *A_subrow,
                                HYPRE_Int       *marker,
                                HYPRE_Int        row_num )
{
   HYPRE_Int      *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex  *A_a = hypre_CSRMatrixData(A);
   HYPRE_Complex  *sub_row_data = hypre_VectorData(A_subrow);

   /* Local variables */
   HYPRE_Int       j, cc;

   for (j = 0; j < hypre_VectorSize(A_subrow); j++)
   {
      sub_row_data[j] = 0.0;
   }

   for (j = A_i[row_num]; j < A_i[row_num + 1]; j++)
   {
      if ((cc = marker[A_j[j]]) >= 0)
      {
         sub_row_data[cc] = A_a[j];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FindKapGrad
 *
 * Finding the Kaporin Gradient contribution (psi) of a given row.
 *
 * Parameters:
 *  - A:            CSR matrix diagonal of A.
 *  - kap_grad:     Array holding the kaporin gradient.
 *                  This will we modified.
 *  - kg_pos:       Array of the nonzero column indices of kap_grad.
 *                  To be modified.
 *  - G_temp:       Work array of G for row i.
 *  - pattern:      Array of column indices of the nonzeros of G_temp.
 *  - patt_size:    Number of column indices of the nonzeros of G_temp.
 *  - max_row_size: To ensure we don't overfill kap_grad.
 *  - row_num:      Which row of G we are working on.
 *  - marker:       Array of length equal to the number of rows in A.
 *                  Assumed to all be set to -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FindKapGrad( hypre_CSRMatrix  *A_diag,
                   hypre_Vector     *kap_grad,
                   HYPRE_Int        *kg_pos,
                   hypre_Vector     *G_temp,
                   HYPRE_Int        *pattern,
                   HYPRE_Int         patt_size,
                   HYPRE_Int         max_row_size,
                   HYPRE_Int         row_num,
                   HYPRE_Int        *kg_marker )
{
   HYPRE_UNUSED_VAR(max_row_size);

   HYPRE_Int      *A_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex  *A_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Complex  *G_temp_data   = hypre_VectorData(G_temp);
   HYPRE_Complex  *kap_grad_data = hypre_VectorData(kap_grad);

   /* Local Variables */
   HYPRE_Int       i, ii, j, k, count, col;

   count = 0;

   /* Compute A[row_num, 0:(row_num-1)]*G_temp[i,i] */
   for (j = A_i[row_num]; j < A_i[row_num + 1]; j++)
   {
      col = A_j[j];
      if (col < row_num)
      {
         if (kg_marker[col] > -1)
         {
            /* Add A[row_num, col] to the tentative pattern */
            kg_marker[col] = count + 1;
            kg_pos[count] = col;
            kap_grad_data[count] = A_a[j];
            count++;
         }
      }
   }

   /* Compute A[0:(row_num-1), P]*G_temp[P, i] */
   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
      {
         col = A_j[j];
         if (col < row_num)
         {
            k = kg_marker[col];
            if (k == 0)
            {
               /* New entry in the tentative pattern */
               kg_marker[col] = count + 1;
               kg_pos[count] = col;
               kap_grad_data[count] = G_temp_data[i] * A_a[j];
               count++;
            }
            else if (k > 0)
            {
               /* Already existing entry in the tentative pattern */
               kap_grad_data[k - 1] += G_temp_data[i] * A_a[j];
            }
         }
      }
   }

   /* Update number of nonzero coefficients held in kap_grad */
   hypre_VectorSize(kap_grad) = count;

   /* Update to absolute values */
   for (i = 0; i < count; i++)
   {
      kap_grad_data[i] = hypre_cabs(kap_grad_data[i]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_swap2_ci
 *--------------------------------------------------------------------------*/

void
hypre_swap2_ci( HYPRE_Complex  *v,
                HYPRE_Int      *w,
                HYPRE_Int       i,
                HYPRE_Int       j )
{
   HYPRE_Complex  temp;
   HYPRE_Int      temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 * hypre_qsort2_ci
 *
 * Quick Sort (largest to smallest) for complex arrays.
 * Sort on real portion of v (HYPRE_Complex), move w.
 *--------------------------------------------------------------------------*/

void
hypre_qsort2_ci( HYPRE_Complex  *v,
                 HYPRE_Int      *w,
                 HYPRE_Int      left,
                 HYPRE_Int      right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }

   hypre_swap2_ci(v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (hypre_creal(v[i]) > hypre_creal(v[left]))
      {
         hypre_swap2_ci(v, w, ++last, i);
      }
   }

   hypre_swap2_ci(v, w, left, last);
   hypre_qsort2_ci(v, w, left, last - 1);
   hypre_qsort2_ci(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 * hypre_PartialSelectSortCI
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_PartialSelectSortCI( HYPRE_Complex  *v,
                           HYPRE_Int      *w,
                           HYPRE_Int       size,
                           HYPRE_Int       nentries )
{
   HYPRE_Int  i, k, pos;

   for (k = 0; k < nentries; k++)
   {
      /* Find largest kth entry */
      pos = k;
      for (i = k + 1; i < size; i++)
      {
         if (hypre_creal(v[i]) > hypre_creal(v[pos]))
         {
            pos = i;
         }
      }

      /* Move entry to beggining of the array */
      hypre_swap2_ci(v, w, k, pos);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AddToPattern
 *
 * Take the largest elements from the kaporin gradient and add their
 * locations to pattern.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AddToPattern( hypre_Vector *kap_grad,
                    HYPRE_Int    *kg_pos,
                    HYPRE_Int    *pattern,
                    HYPRE_Int    *patt_size,
                    HYPRE_Int    *kg_marker,
                    HYPRE_Int     max_step_size )
{
   HYPRE_Int       kap_grad_size = hypre_VectorSize(kap_grad);
   HYPRE_Complex  *kap_grad_data = hypre_VectorData(kap_grad);

   HYPRE_Int       i, nentries;

   /* Number of entries that can be added */
   nentries = hypre_min(kap_grad_size, max_step_size);

   /* Reorder candidates according to larger weights */
   //hypre_qsort2_ci(kap_grad_data, &kg_pos, 0, kap_grad_size-1);
   hypre_PartialSelectSortCI(kap_grad_data, kg_pos, kap_grad_size, nentries);

   /* Update pattern with new entries */
   for (i = 0; i < nentries; i++)
   {
      pattern[*patt_size + i] = kg_pos[i];
   }
   *patt_size += nentries;

   /* Put pattern in ascending order */
   hypre_qsort0(pattern, 0, (*patt_size) - 1);

   /* Reset marked entries that are added to pattern */
   for (i = 0; i < nentries; i++)
   {
      kg_marker[kg_pos[i]] = -1;
   }
   for (i = nentries; i < kap_grad_size; i++)
   {
      kg_marker[kg_pos[i]] = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseSPDSystemSolve
 *
 * Solve the dense SPD linear system with LAPACK:
 *
 *    mat*lhs = -rhs
 *
 * Note: the contents of A change to its Cholesky factor.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseSPDSystemSolve( hypre_Vector *mat,
                           hypre_Vector *rhs,
                           hypre_Vector *lhs )
{
   HYPRE_Int      size = hypre_VectorSize(rhs);
   HYPRE_Complex *mat_data = hypre_VectorData(mat);
   HYPRE_Complex *rhs_data = hypre_VectorData(rhs);
   HYPRE_Complex *lhs_data = hypre_VectorData(lhs);

   /* Local variables */
   HYPRE_Int      num_rhs = 1;
   char           uplo = 'L';
   char           msg[512];
   HYPRE_Int      i, info;

   /* Copy RHS into LHS */
   for (i = 0; i < size; i++)
   {
      lhs_data[i] = -rhs_data[i];
   }

   /* Compute Cholesky factor */
   hypre_dpotrf(&uplo, &size, mat_data, &size, &info);
   if (info)
   {
      hypre_sprintf(msg, "Error: dpotrf failed with code %d\n", info);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   /* Solve dense linear system */
   hypre_dpotrs(&uplo, &size, &num_rhs, mat_data, &size, lhs_data, &size, &info);
   if (info)
   {
      hypre_sprintf(msg, "Error: dpotrs failed with code %d\n", info);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupNative
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupNative( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex          *A_a              = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);

   /* Matrix G variables */
   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag;
   HYPRE_Int              *G_i;
   HYPRE_Int              *G_j;
   HYPRE_Complex          *G_a;
   HYPRE_Int               max_nnzrow_diag_G;   /* Max. number of nonzeros per row in G_diag */

   /* Local variables */
   char                    msg[512];    /* Warning message */
   HYPRE_Int              *twspace;     /* shared work space for omp threads */

   /* Initalize some variables */
   max_nnzrow_diag_G = max_steps * max_step_size + 1;

   G_diag = hypre_ParCSRMatrixDiag(G);
   G_a = hypre_CSRMatrixData(G_diag);
   G_i = hypre_CSRMatrixI(G_diag);
   G_j = hypre_CSRMatrixJ(G_diag);

   /* Allocate shared work space array for OpenMP threads */
   twspace = hypre_CTAlloc(HYPRE_Int, hypre_NumThreads() + 1, HYPRE_MEMORY_HOST);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      hypre_Vector   *G_temp;        /* Vector holding the values of G[i,:] */
      hypre_Vector   *A_sub;         /* Vector holding the dense submatrix A[P, P] */
      hypre_Vector   *A_subrow;      /* Vector holding A[i, P] */
      hypre_Vector   *kap_grad;      /* Vector holding the Kaporin gradient values */
      HYPRE_Int      *kg_pos;        /* Indices of nonzero entries of kap_grad */
      HYPRE_Int      *kg_marker;     /* Marker array with nonzeros pointing to kg_pos */
      HYPRE_Int      *marker;        /* Marker array with nonzeros pointing to P */
      HYPRE_Int      *pattern;       /* Array holding column indices of G[i,:] */
      HYPRE_Int       patt_size;     /* Number of entries in current pattern */
      HYPRE_Int       patt_size_old; /* Number of entries in previous pattern */
      HYPRE_Int       ii;            /* Thread identifier */
      HYPRE_Int       num_threads;   /* Number of active threads */
      HYPRE_Int       ns, ne;        /* Initial and last row indices */
      HYPRE_Int       i, j, k, iloc; /* Loop variables */
      HYPRE_Complex   old_psi;       /* GAG' before k-th interation of aFSAI */
      HYPRE_Complex   new_psi;       /* GAG' after k-th interation of aFSAI */
      HYPRE_Complex   row_scale;     /* Scaling factor for G_temp */
      HYPRE_Complex  *G_temp_data;
      HYPRE_Complex  *A_subrow_data;

      HYPRE_Int       num_rows_Gloc;
      HYPRE_Int       num_nnzs_Gloc;
      HYPRE_Int      *Gloc_i;
      HYPRE_Int      *Gloc_j;
      HYPRE_Complex  *Gloc_a;

      /* Allocate and initialize local vector variables */
      G_temp    = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      A_subrow  = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      kap_grad  = hypre_SeqVectorCreate(num_rows_diag_A);
      A_sub     = hypre_SeqVectorCreate(max_nnzrow_diag_G * max_nnzrow_diag_G);
      pattern   = hypre_CTAlloc(HYPRE_Int, max_nnzrow_diag_G, HYPRE_MEMORY_HOST);
      kg_pos    = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
      kg_marker = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
      marker    = hypre_TAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

      hypre_SeqVectorInitialize_v2(G_temp, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_subrow, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(kap_grad, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_sub, HYPRE_MEMORY_HOST);
      hypre_Memset(marker, -1, (size_t) num_rows_diag_A * sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

      /* Setting data variables for vectors */
      G_temp_data   = hypre_VectorData(G_temp);
      A_subrow_data = hypre_VectorData(A_subrow);

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      hypre_partition1D(num_rows_diag_A, num_threads, ii, &ns, &ne);

      num_rows_Gloc = ne - ns;
      if (num_threads == 1)
      {
         Gloc_i = G_i;
         Gloc_j = G_j;
         Gloc_a = G_a;
      }
      else
      {
         num_nnzs_Gloc = num_rows_Gloc * max_nnzrow_diag_G;

         Gloc_i = hypre_CTAlloc(HYPRE_Int, num_rows_Gloc + 1, HYPRE_MEMORY_HOST);
         Gloc_j = hypre_CTAlloc(HYPRE_Int, num_nnzs_Gloc, HYPRE_MEMORY_HOST);
         Gloc_a = hypre_CTAlloc(HYPRE_Complex, num_nnzs_Gloc, HYPRE_MEMORY_HOST);
      }

      for (i = ns; i < ne; i++)
      {
         patt_size = 0;

         /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
         new_psi = old_psi = A_a[A_i[i]];

         /* Cycle through each iteration for that row */
         for (k = 0; k < max_steps; k++)
         {
            /* Compute Kaporin Gradient */
            hypre_FindKapGrad(A_diag, kap_grad, kg_pos, G_temp, pattern,
                              patt_size, max_nnzrow_diag_G, i, kg_marker);

            /* Find max_step_size largest values of the kaporin gradient,
               find their column indices, and add it to pattern */
            patt_size_old = patt_size;
            hypre_AddToPattern(kap_grad, kg_pos, pattern, &patt_size,
                               kg_marker, max_step_size);

            /* Update sizes */
            hypre_VectorSize(A_sub)    = patt_size * patt_size;
            hypre_VectorSize(A_subrow) = patt_size;
            hypre_VectorSize(G_temp)   = patt_size;

            if (patt_size == patt_size_old)
            {
               new_psi = old_psi;
               break;
            }
            else
            {
               /* Gather A[P, P] and -A[i, P] */
               for (j = 0; j < patt_size; j++)
               {
                  marker[pattern[j]] = j;
               }
               hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker);
               hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);

               /* Solve A[P, P] G[i, P]' = -A[i, P] */
               hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

               /* Determine psi_{k+1} = G_temp[i] * A[P, P] * G_temp[i]' */
               new_psi = A_a[A_i[i]];
               for (j = 0; j < patt_size; j++)
               {
                  new_psi += G_temp_data[j] * A_subrow_data[j];
               }

               /* Check psi reduction */
               if (hypre_cabs(new_psi - old_psi) < hypre_creal(kap_tolerance * old_psi))
               {
                  break;
               }
               else
               {
                  old_psi = new_psi;
               }
            }
         }

         /* Reset marker for building dense linear system */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = -1;
         }

         /* Compute scaling factor */
         if (hypre_creal(new_psi) > 0 && hypre_cimag(new_psi) == 0)
         {
            row_scale = 1.0 / hypre_csqrt(new_psi);
         }
         else
         {
            hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);

            row_scale = 1.0 / hypre_cabs(A_a[A_i[i]]);
            hypre_VectorSize(G_temp) = patt_size = 0;
         }

         /* Pass values of G_temp into G */
         iloc = i - ns;
         Gloc_j[Gloc_i[iloc]] = i;
         Gloc_a[Gloc_i[iloc]] = row_scale;
         for (k = 0; k < patt_size; k++)
         {
            j = Gloc_i[iloc] + k + 1;
            Gloc_j[j] = pattern[k];
            Gloc_a[j] = row_scale * G_temp_data[k];
            kg_marker[pattern[k]] = 0;
         }
         Gloc_i[iloc + 1] = Gloc_i[iloc] + k + 1;
      }

      /* Copy data to shared memory */
      twspace[ii + 1] = Gloc_i[num_rows_Gloc] - Gloc_i[0];
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
      #pragma omp single
#endif
      {
         for (i = 0; i < num_threads; i++)
         {
            twspace[i + 1] += twspace[i];
         }
      }

      if (num_threads > 1)
      {
         /* Correct row pointer G_i */
         G_i[ns] = twspace[ii];
         for (i = ns; i < ne; i++)
         {
            iloc = i - ns;
            G_i[i + 1] = G_i[i] + Gloc_i[iloc + 1] - Gloc_i[iloc];
         }

         /* Move G_j and G_a */
         for (i = ns; i < ne; i++)
         {
            for (j = G_i[i]; j < G_i[i + 1]; j++)
            {
               G_j[j] = Gloc_j[j - G_i[ns]];
               G_a[j] = Gloc_a[j - G_i[ns]];
            }
         }

         hypre_TFree(Gloc_i, HYPRE_MEMORY_HOST);
         hypre_TFree(Gloc_j, HYPRE_MEMORY_HOST);
         hypre_TFree(Gloc_a, HYPRE_MEMORY_HOST);
      }

      /* Free memory */
      hypre_SeqVectorDestroy(G_temp);
      hypre_SeqVectorDestroy(A_subrow);
      hypre_SeqVectorDestroy(kap_grad);
      hypre_SeqVectorDestroy(A_sub);
      hypre_TFree(kg_pos, HYPRE_MEMORY_HOST);
      hypre_TFree(pattern, HYPRE_MEMORY_HOST);
      hypre_TFree(marker, HYPRE_MEMORY_HOST);
      hypre_TFree(kg_marker, HYPRE_MEMORY_HOST);
   } /* end openmp region */
   HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);

   /* Update local number of nonzeros of G */
   hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupOMPDyn
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupOMPDyn( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex          *A_a              = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);

   /* Matrix G variables */
   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag;
   HYPRE_Int              *G_i;
   HYPRE_Int              *G_j;
   HYPRE_Complex          *G_a;
   HYPRE_Int              *G_nnzcnt;          /* Array holding number of nonzeros of row G[i,:] */
   HYPRE_Int               max_nnzrow_diag_G; /* Max. number of nonzeros per row in G_diag */

   /* Local variables */
   HYPRE_Int               iG, jG, jjG;
   char                    msg[512];    /* Warning message */
   HYPRE_Complex          *twspace;     /* shared work space for omp threads */

   /* Initalize some variables */
   max_nnzrow_diag_G = max_steps * max_step_size + 1;

   G_diag = hypre_ParCSRMatrixDiag(G);
   G_a = hypre_CSRMatrixData(G_diag);
   G_i = hypre_CSRMatrixI(G_diag);
   G_j = hypre_CSRMatrixJ(G_diag);
   G_nnzcnt = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

   /* Allocate shared work space array for OpenMP threads */
   twspace = hypre_CTAlloc(HYPRE_Complex, hypre_NumThreads() + 1, HYPRE_MEMORY_HOST);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      hypre_Vector   *G_temp;        /* Vector holding the values of G[i,:] */
      hypre_Vector   *A_sub;         /* Vector holding the dense submatrix A[P, P] */
      hypre_Vector   *A_subrow;      /* Vector holding A[i, P] */
      hypre_Vector   *kap_grad;      /* Vector holding the Kaporin gradient values */
      HYPRE_Int      *kg_pos;        /* Indices of nonzero entries of kap_grad */
      HYPRE_Int      *kg_marker;     /* Marker array with nnzs pointing to kg_pos */
      HYPRE_Int      *marker;        /* Marker array with nonzeros pointing to P */
      HYPRE_Int      *pattern;       /* Array holding column indices of G[i,:] */
      HYPRE_Int       patt_size;     /* Number of entries in current pattern */
      HYPRE_Int       patt_size_old; /* Number of entries in previous pattern */
      HYPRE_Int       i, j, k;       /* Loop variables */
      HYPRE_Complex   old_psi;       /* GAG' before k-th interation of aFSAI */
      HYPRE_Complex   new_psi;       /* GAG' after k-th interation of aFSAI */
      HYPRE_Complex   row_scale;     /* Scaling factor for G_temp */
      HYPRE_Complex  *G_temp_data;
      HYPRE_Complex  *A_subrow_data;

      /* Allocate and initialize local vector variables */
      G_temp    = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      A_subrow  = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      kap_grad  = hypre_SeqVectorCreate(num_rows_diag_A);
      A_sub     = hypre_SeqVectorCreate(max_nnzrow_diag_G * max_nnzrow_diag_G);
      pattern   = hypre_CTAlloc(HYPRE_Int, max_nnzrow_diag_G, HYPRE_MEMORY_HOST);
      kg_pos    = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
      kg_marker = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
      marker    = hypre_TAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

      hypre_SeqVectorInitialize_v2(G_temp, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_subrow, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(kap_grad, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_sub, HYPRE_MEMORY_HOST);
      hypre_Memset(marker, -1, (size_t) num_rows_diag_A * sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

      /* Setting data variables for vectors */
      G_temp_data   = hypre_VectorData(G_temp);
      A_subrow_data = hypre_VectorData(A_subrow);

#ifdef HYPRE_USING_OPENMP
      #pragma omp for schedule(dynamic)
#endif
      for (i = 0; i < num_rows_diag_A; i++)
      {
         patt_size = 0;

         /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
         new_psi = old_psi = A_a[A_i[i]];

         /* Cycle through each iteration for that row */
         for (k = 0; k < max_steps; k++)
         {
            /* Compute Kaporin Gradient */
            hypre_FindKapGrad(A_diag, kap_grad, kg_pos, G_temp, pattern,
                              patt_size, max_nnzrow_diag_G, i, kg_marker);

            /* Find max_step_size largest values of the kaporin gradient,
               find their column indices, and add it to pattern */
            patt_size_old = patt_size;
            hypre_AddToPattern(kap_grad, kg_pos, pattern, &patt_size,
                               kg_marker, max_step_size);

            /* Update sizes */
            hypre_VectorSize(A_sub)    = patt_size * patt_size;
            hypre_VectorSize(A_subrow) = patt_size;
            hypre_VectorSize(G_temp)   = patt_size;

            if (patt_size == patt_size_old)
            {
               new_psi = old_psi;
               break;
            }
            else
            {
               /* Gather A[P, P] and -A[i, P] */
               for (j = 0; j < patt_size; j++)
               {
                  marker[pattern[j]] = j;
               }
               hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker);
               hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);

               /* Solve A[P, P] G[i, P]' = -A[i, P] */
               hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

               /* Determine psi_{k+1} = G_temp[i] * A[P, P] * G_temp[i]' */
               new_psi = A_a[A_i[i]];
               for (j = 0; j < patt_size; j++)
               {
                  new_psi += G_temp_data[j] * A_subrow_data[j];
               }

               /* Check psi reduction */
               if (hypre_cabs(new_psi - old_psi) < hypre_creal(kap_tolerance * old_psi))
               {
                  break;
               }
               else
               {
                  old_psi = new_psi;
               }
            }
         }

         /* Reset marker for building dense linear system */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = -1;
         }

         /* Compute scaling factor */
         if (hypre_creal(new_psi) > 0 && hypre_cimag(new_psi) == 0)
         {
            row_scale = 1.0 / hypre_csqrt(new_psi);
         }
         else
         {
            hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);

            row_scale = 1.0 / hypre_cabs(A_a[A_i[i]]);
            hypre_VectorSize(G_temp) = patt_size = 0;
         }

         /* Pass values of G_temp into G */
         j = i * max_nnzrow_diag_G;
         G_j[j] = i;
         G_a[j] = row_scale;
         j++;
         for (k = 0; k < patt_size; k++)
         {
            G_j[j] = pattern[k];
            G_a[j++] = row_scale * G_temp_data[k];
            kg_marker[pattern[k]] = 0;
         }
         G_nnzcnt[i] = patt_size + 1;
      } /* omp for schedule(dynamic) */

      /* Free memory */
      hypre_SeqVectorDestroy(G_temp);
      hypre_SeqVectorDestroy(A_subrow);
      hypre_SeqVectorDestroy(kap_grad);
      hypre_SeqVectorDestroy(A_sub);
      hypre_TFree(kg_pos, HYPRE_MEMORY_HOST);
      hypre_TFree(pattern, HYPRE_MEMORY_HOST);
      hypre_TFree(marker, HYPRE_MEMORY_HOST);
      hypre_TFree(kg_marker, HYPRE_MEMORY_HOST);
   } /* end openmp region */
   HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Reorder array */
   G_i[0] = 0;
   for (iG = 0; iG < num_rows_diag_A; iG++)
   {
      G_i[iG + 1] = G_i[iG] + G_nnzcnt[iG];
      jjG = iG * max_nnzrow_diag_G;
      for (jG = G_i[iG]; jG < G_i[iG + 1]; jG++)
      {
         G_j[jG] = G_j[jjG];
         G_a[jG] = G_a[jjG++];
      }
   }

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);
   hypre_TFree(G_nnzcnt, HYPRE_MEMORY_HOST);

   /* Update local number of nonzeros of G */
   hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BitFSAIFindPosHost
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_BitFSAIFindPosHost( const HYPRE_Int *A_i,
                          const HYPRE_Int *A_j,
                          HYPRE_Int        row,
                          HYPRE_Int        target_col )
{
   HYPRE_Int k;
   for (k = A_i[row]; k < A_i[row + 1]; k++)
   {
      if (A_j[k] == target_col)
      {
         return k;
      }
   }
   return -1;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupBitFSAIHost
 *
 * Zero-duplicate-index 4-bit split-scale BitFSAI setup on host.
 * Computes local lower-triangular Cholesky patches directly on A_diag,
 * quantizes off-diagonal entries to symmetric signed INT4 nibbles, applies
 * 2x2 A-energy re-projection, and mirrors nibbles onto A_diag's CSR slots.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_BitFSAIIsFiniteHost( double v )
{
   return (v == v && v >= -DBL_MAX && v <= DBL_MAX);
}

static HYPRE_Int
hypre_BitFSAIIsStorableF32Host( double v )
{
   float s = (float) v;
   return (hypre_BitFSAIIsFiniteHost(v) && s == s && s != 0.0f && s >= -FLT_MAX && s <= FLT_MAX);
}

static HYPRE_Int
hypre_BitFSAIProjectRowHost( HYPRE_Int            m,
                             const double        *q_dec,
                             const double        *L,
                             const double        *A_patch,
                             double              *diag_out,
                             double              *scale_out )
{
   HYPRE_Int a, b;
   double c11 = 0.0;
   double c12 = 0.0;
   double projection;
   double residual_sq = 0.0;
   double g_opt, s_opt;
   float g_f32, s_f32;
   double sg, ss;
   double energy = 0.0;
   double comp = 0.0;
   double sum_abs = 0.0;
   double bound;

   if (m < 2)
   {
      return 0;
   }
   if (!(A_patch[(m - 1) * 16 + (m - 1)] > 0.0) ||
       !hypre_BitFSAIIsFiniteHost(A_patch[(m - 1) * 16 + (m - 1)]))
   {
      return -1;
   }

   for (b = 0; b < m - 1; b++)
   {
      double ub = 0.0;
      for (a = b; a < m - 1; a++)
      {
         ub += L[a * 16 + b] * q_dec[a];
      }
      c11 += ub * ub;
      c12 += ub * L[(m - 1) * 16 + b];
   }
   if (!(c11 > 0.0) || !hypre_BitFSAIIsFiniteHost(c11) || !hypre_BitFSAIIsFiniteHost(c12))
   {
      return 0;
   }

   projection = c12 / c11;
   if (!hypre_BitFSAIIsFiniteHost(projection))
   {
      return 0;
   }

   /* Orthogonal residual avoids cancellation in c11 * c22 - c12^2 */
   for (b = 0; b < m; b++)
   {
      double ub = 0.0;
      double res;
      if (b < m - 1)
      {
         for (a = b; a < m - 1; a++)
         {
            ub += L[a * 16 + b] * q_dec[a];
         }
      }
      res = L[(m - 1) * 16 + b] - projection * ub;
      residual_sq += res * res;
   }
   if (!(residual_sq > 0.0) || !hypre_BitFSAIIsFiniteHost(residual_sq))
   {
      return 0;
   }

   g_opt = 1.0 / sqrt(residual_sq);
   s_opt = -projection * g_opt;
   if (!(g_opt > 0.0) || !hypre_BitFSAIIsFiniteHost(g_opt) || !hypre_BitFSAIIsFiniteHost(s_opt))
   {
      return 0;
   }

   g_f32 = (float) g_opt;
   s_f32 = (float) s_opt;
   if (s_f32 == 0.0f)
   {
      return 0;
   }
   if (!hypre_BitFSAIIsStorableF32Host(g_opt) || !hypre_BitFSAIIsStorableF32Host(s_opt))
   {
      return -2;
   }

   sg = (double) g_f32;
   ss = (double) s_f32;
   for (a = 0; a < m; a++)
   {
      double wa = (a == m - 1) ? sg : (ss * q_dec[a]);
      for (b = 0; b < m; b++)
      {
         double wb = (b == m - 1) ? sg : (ss * q_dec[b]);
         double a_ab = (a >= b) ? A_patch[a * 16 + b] : A_patch[b * 16 + a];
         double term = wa * a_ab * wb;
         double adj, next;
         if (!hypre_BitFSAIIsFiniteHost(term))
         {
            return -2;
         }
         adj = term - comp;
         next = energy + adj;
         comp = (next - energy) - adj;
         energy = next;
         sum_abs += fabs(term);
      }
   }
   if (!hypre_BitFSAIIsFiniteHost(energy) || !hypre_BitFSAIIsFiniteHost(sum_abs))
   {
      return -2;
   }
   bound = 8.0 * DBL_EPSILON * sum_abs;
   if (!(fabs(energy - 1.0) + bound <= 1.0e-4))
   {
      return 0;
   }

   *diag_out  = g_opt;
   *scale_out = s_opt;
   return 1;
}

static HYPRE_Int
hypre_FSAISetupBitFSAIHost( void               *fsai_vdata,
                            hypre_ParCSRMatrix *A )
{
#if defined(HYPRE_COMPLEX)
   HYPRE_UNUSED_VAR(fsai_vdata);
   HYPRE_UNUSED_VAR(A);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "BitFSAI (algo_type = 4) does not support complex arithmetic\n");
   return hypre_error_flag;
#else
   hypre_ParFSAIData *fsai_data   = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int          max_nnz_row = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   hypre_CSRMatrix   *A_diag      = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int          num_rows    = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int          nnz         = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int         *A_i         = hypre_CSRMatrixI(A_diag);
   HYPRE_Int         *A_j         = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex     *A_a         = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          patch_cap;
   HYPRE_Int          packed_bytes;
   HYPRE_Int          i;
   HYPRE_Int          err_invalid_diag = 0;
   HYPRE_Int          err_unrep        = 0;
   HYPRE_Int          err_nonsym       = 0;
   HYPRE_Int          wide_apply       = 0;
   float             *g_diag;
   float             *g_scale;
   unsigned char     *q_tmp;
   unsigned char     *q_packed;
   float             *y_work = NULL;
   double            *t_work = NULL;

   hypre_FSAIFreeBitFSAI(fsai_vdata);

   hypre_ParFSAIDataBitFSAIADiag(fsai_data)   = A_diag;
   hypre_ParFSAIDataBitFSAINumRows(fsai_data) = num_rows;
   hypre_ParFSAIDataBitFSAINnz(fsai_data)     = nnz;
   hypre_ParFSAIDataBitFSAIMemLoc(fsai_data)  = HYPRE_MEMORY_HOST;

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
   g_diag   = hypre_TAlloc(float, num_rows, HYPRE_MEMORY_HOST);
   g_scale  = hypre_TAlloc(float, num_rows, HYPRE_MEMORY_HOST);
   q_packed = hypre_TAlloc(unsigned char, (packed_bytes > 0) ? packed_bytes : 1, HYPRE_MEMORY_HOST);
   q_tmp    = hypre_TAlloc(unsigned char, (nnz > 0) ? nnz : 1, HYPRE_MEMORY_HOST);

   if (nnz > 0)
   {
      hypre_Memset(q_tmp, 8, (size_t) nnz * sizeof(unsigned char), HYPRE_MEMORY_HOST);
   }

#if !defined(HYPRE_SINGLE)
   wide_apply = 1;
#endif

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for schedule(guided) private(i) reduction(|:err_invalid_diag,err_unrep,err_nonsym,wide_apply)
#endif
   for (i = 0; i < num_rows; i++)
   {
      HYPRE_Int     J[16];
      HYPRE_Int     Klow[16];
      HYPRE_Int     Kup[16];
      double        a_off[16];
      double        A_patch[16 * 16];
      double        L[16 * 16];
      double        g_off[16];
      double        q_dec[16];
      unsigned char q_code[16];
      HYPRE_Int     rs = A_i[i];
      HYPRE_Int     re = A_i[i + 1];
      HYPRE_Int     k, a, b, c;
      HYPRE_Int     m_off = 0;
      HYPRE_Int     m;
      HYPRE_Int     ok = 1;
      HYPRE_Int     found_diag = 0;
      HYPRE_Int     diagonal_only = 0;
      double        a_ii = 0.0;
      double        g_d = 0.0;
      double        scale = 0.0;

      for (k = rs; k < re; k++)
      {
         HYPRE_Int col = A_j[k];
         double    val = (double) A_a[k];
         if (col == i)
         {
            a_ii = val;
            found_diag = 1;
         }
         else if (col < i)
         {
            HYPRE_Int k_up = hypre_BitFSAIFindPosHost(A_i, A_j, col, i);
            if (k_up < 0)
            {
               err_nonsym = 1;
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
               for (a = 1; a < m_off; a++)
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
            if (hypre_BitFSAIFindPosHost(A_i, A_j, col, i) < 0)
            {
               err_nonsym = 1;
            }
         }
      }

      if (!found_diag || !(a_ii > 0.0) || !hypre_BitFSAIIsFiniteHost(a_ii))
      {
         err_invalid_diag = 1;
         g_diag[i]  = 1.0f;
         g_scale[i] = 1.0f;
         continue;
      }

      if (m_off == 0)
      {
         diagonal_only = 1;
         g_d   = 1.0 / sqrt(a_ii);
         scale = g_d;
      }
      else
      {
         m = m_off + 1;
         for (a = 0; a < m; a++)
         {
            for (b = 0; b <= a; b++)
            {
               A_patch[a * 16 + b] = 0.0;
            }
         }

         for (a = 0; a < m_off; a++)
         {
            HYPRE_Int ra = J[a];
            for (k = A_i[ra]; k < A_i[ra + 1]; k++)
            {
               HYPRE_Int col = A_j[k];
               for (b = 0; b <= a; b++)
               {
                  if (J[b] == col)
                  {
                     A_patch[a * 16 + b] = (double) A_a[k];
                     break;
                  }
               }
            }
            A_patch[(m - 1) * 16 + a] = a_off[a];
         }
         A_patch[(m - 1) * 16 + (m - 1)] = a_ii;

         for (a = 0; a < m; a++)
         {
            for (b = 0; b <= a; b++)
            {
               L[a * 16 + b] = A_patch[a * 16 + b];
            }
         }

         /* Cholesky factorization L L^T = A_patch */
         for (a = 0; a < m && ok; a++)
         {
            for (b = 0; b <= a; b++)
            {
               double s = L[a * 16 + b];
               for (c = 0; c < b; c++)
               {
                  s -= L[a * 16 + c] * L[b * 16 + c];
               }
               if (a == b)
               {
                  if (!(s > 0.0) || !hypre_BitFSAIIsFiniteHost(s))
                  {
                     ok = 0;
                     break;
                  }
                  L[a * 16 + a] = sqrt(s);
               }
               else
               {
                  double lv = s / L[b * 16 + b];
                  if (!hypre_BitFSAIIsFiniteHost(lv))
                  {
                     ok = 0;
                     break;
                  }
                  L[a * 16 + b] = lv;
               }
            }
         }

         if (ok)
         {
            g_d = 1.0 / L[(m - 1) * 16 + (m - 1)];
            if (!hypre_BitFSAIIsFiniteHost(g_d))
            {
               ok = 0;
            }
            else
            {
               for (a = m_off - 1; a >= 0; a--)
               {
                  double s = -L[(m - 1) * 16 + a] * g_d;
                  for (b = a + 1; b < m_off; b++)
                  {
                     s -= L[b * 16 + a] * g_off[b];
                  }
                  g_off[a] = s / L[a * 16 + a];
                  if (!hypre_BitFSAIIsFiniteHost(g_off[a]))
                  {
                     ok = 0;
                     break;
                  }
               }
            }
         }

         if (!ok)
         {
            diagonal_only = 1;
            g_d   = 1.0 / sqrt(a_ii);
            scale = g_d;
         }
         else
         {
            scale = 0.0;
            for (a = 0; a < m_off; a++)
            {
               double av = fabs(g_off[a]);
               if (av > scale)
               {
                  scale = av;
               }
            }

            if (!(scale > 0.0) || !hypre_BitFSAIIsFiniteHost(scale))
            {
               diagonal_only = 1;
               g_d   = 1.0 / sqrt(a_ii);
               scale = g_d;
            }
            else
            {
               for (a = 0; a < m_off; a++)
               {
                  double    u = 7.0 * (g_off[a] / scale);
                  HYPRE_Int q = (HYPRE_Int)(u >= 0.0 ? floor(u + 0.5) : ceil(u - 0.5));
                  if (q > 7)  { q = 7; }
                  if (q < -7) { q = -7; }
                  q_code[a] = (unsigned char)(q + 8);
                  q_dec[a]  = (double)(((float) q) * (1.0f / 7.0f));
               }

               {
                  HYPRE_Int proj_rc = hypre_BitFSAIProjectRowHost(m, q_dec, L, A_patch, &g_d, &scale);
                  if (proj_rc == -1)
                  {
                     err_invalid_diag = 1;
                     g_diag[i]  = 1.0f;
                     g_scale[i] = 1.0f;
                     continue;
                  }
                  if (proj_rc == -2)
                  {
                     err_unrep = 1;
                     g_diag[i]  = 1.0f;
                     g_scale[i] = 1.0f;
                     continue;
                  }
                  if (proj_rc == 0)
                  {
                     diagonal_only = 1;
                     g_d   = 1.0 / sqrt(a_ii);
                     scale = g_d;
                     for (a = 0; a < m_off; a++)
                     {
                        q_code[a] = 8;
                     }
                  }
                  else
                  {
                     for (a = 0; a < m_off; a++)
                     {
                        q_tmp[Klow[a]] = q_code[a];
                        q_tmp[Kup[a]]  = q_code[a];
                     }
                  }
               }
            }
         }
      }

      if (!hypre_BitFSAIIsStorableF32Host(g_d) || !hypre_BitFSAIIsStorableF32Host(scale))
      {
         err_unrep = 1;
         g_diag[i]  = 1.0f;
         g_scale[i] = 1.0f;
         continue;
      }
      if (diagonal_only)
      {
         double sd = (double)(float) g_d;
         double se = sd * sd * a_ii;
         if (!hypre_BitFSAIIsFiniteHost(se) || fabs(se - 1.0) > 1.0e-4)
         {
            err_unrep = 1;
            g_diag[i]  = 1.0f;
            g_scale[i] = 1.0f;
            continue;
         }
      }

      {
         double ratio = fabs(scale) / g_d;
         if (ratio < 1e-6 || ratio > 1e6 ||
             fabs(scale) < sqrt((double) FLT_MIN) ||
             fabs(scale) > sqrt((double) FLT_MAX))
         {
            wide_apply = 1;
         }
      }

      g_diag[i]  = (float) g_d;
      g_scale[i] = (float) scale;
   }

   for (i = 0; i < packed_bytes; i++)
   {
      unsigned char lo = q_tmp[2 * i] & 0x0F;
      unsigned char hi = (2 * i + 1 < nnz) ? (q_tmp[2 * i + 1] & 0x0F) : 8;
      q_packed[i] = (unsigned char)(lo | (hi << 4));
   }

   hypre_TFree(q_tmp, HYPRE_MEMORY_HOST);

   if (err_invalid_diag || err_unrep || err_nonsym)
   {
      hypre_TFree(g_diag, HYPRE_MEMORY_HOST);
      hypre_TFree(g_scale, HYPRE_MEMORY_HOST);
      hypre_TFree(q_packed, HYPRE_MEMORY_HOST);
      hypre_ParFSAIDataBitFSAIADiag(fsai_data)   = NULL;
      hypre_ParFSAIDataBitFSAINumRows(fsai_data) = 0;
      hypre_ParFSAIDataBitFSAINnz(fsai_data)     = 0;
      if (err_nonsym)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "BitFSAI (algo_type = 4) requires structurally symmetric (mirrored) A_diag\n");
      }
      else if (err_invalid_diag)
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

   if (wide_apply)
   {
      t_work = hypre_TAlloc(double, num_rows, HYPRE_MEMORY_HOST);
   }
   else
   {
      y_work = hypre_TAlloc(float, num_rows, HYPRE_MEMORY_HOST);
   }

   hypre_ParFSAIDataBitFSAIWideApply(fsai_data) = wide_apply;
   hypre_ParFSAIDataBitFSAIGDiag(fsai_data)     = g_diag;
   hypre_ParFSAIDataBitFSAIGScale(fsai_data)    = g_scale;
   hypre_ParFSAIDataBitFSAIQPacked(fsai_data)   = q_packed;
   hypre_ParFSAIDataBitFSAIYWork(fsai_data)     = y_work;
   hypre_ParFSAIDataBitFSAITWork(fsai_data)     = t_work;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetup( void               *fsai_vdata,
                 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *f,
                 hypre_ParVector    *u )
{
   hypre_ParFSAIData       *fsai_data     = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int                max_steps     = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int                max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int                max_nnz_row   = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   HYPRE_Int                algo_type     = hypre_ParFSAIDataAlgoType(fsai_data);
   HYPRE_Int                print_level   = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int                eig_max_iters = hypre_ParFSAIDataEigMaxIters(fsai_data);

   /* ParCSRMatrix A variables */
   MPI_Comm                 comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt             num_rows_A    = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt             num_cols_A    = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt            *row_starts_A  = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt            *col_starts_A  = hypre_ParCSRMatrixColStarts(A);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix         *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);

   /* Work vectors */
   hypre_ParVector         *r_work;
   hypre_ParVector         *z_work;

   /* G variables */
   hypre_ParCSRMatrix      *G;
   HYPRE_Int                max_nnzrow_diag_G;   /* Max. number of nonzeros per row in G_diag */
   HYPRE_Int                max_nonzeros_diag_G; /* Max. number of nonzeros in G_diag */

   /* Sanity check */
   if (f && hypre_ParVectorNumVectors(f) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "FSAI doesn't support multicomponent vectors");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SolverResetIsSetup((hypre_Solver *) fsai_vdata);

   /* Release any previously owned setup buffers to avoid leaks on re-setup */
   if (hypre_ParFSAIDataRWork(fsai_data))
   {
      hypre_ParVectorDestroy(hypre_ParFSAIDataRWork(fsai_data));
      hypre_ParFSAIDataRWork(fsai_data) = NULL;
   }
   if (hypre_ParFSAIDataZWork(fsai_data))
   {
      hypre_ParVectorDestroy(hypre_ParFSAIDataZWork(fsai_data));
      hypre_ParFSAIDataZWork(fsai_data) = NULL;
   }
   if (hypre_ParFSAIDataGmat(fsai_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParFSAIDataGmat(fsai_data));
      hypre_ParFSAIDataGmat(fsai_data) = NULL;
   }
   if (hypre_ParFSAIDataGTmat(fsai_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParFSAIDataGTmat(fsai_data));
      hypre_ParFSAIDataGTmat(fsai_data) = NULL;
   }
   hypre_FSAIFreeBitFSAI(fsai_vdata);

   /* Create and initialize work vectors used in the solve phase */
   r_work = hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);
   hypre_ParVectorInitialize(r_work);
   hypre_ParFSAIDataRWork(fsai_data) = r_work;

   if (algo_type == 4)
   {
      HYPRE_Int local_err = 0;
      HYPRE_Int global_err = 0;

      hypre_ParFSAIDataZWork(fsai_data) = NULL;
      hypre_ParFSAIDataGmat(fsai_data)  = NULL;
      hypre_ParFSAIDataGTmat(fsai_data) = NULL;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      HYPRE_MemoryLocation  memloc_A = hypre_ParCSRMatrixMemoryLocation(A);
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memloc_A);

      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_FSAISetupBitFSAIDevice(fsai_vdata, A);
      }
      else
#elif defined(HYPRE_USING_GPU)
      HYPRE_MemoryLocation  memloc_A = hypre_ParCSRMatrixMemoryLocation(A);
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memloc_A);

      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "BitFSAI (algo_type = 4) device execution requires CUDA or HIP\n");
      }
      else
#endif
      {
         hypre_FSAISetupBitFSAIHost(fsai_vdata, A);
      }

      local_err = (hypre_error_flag != 0) ? 1 : 0;
      hypre_MPI_Allreduce(&local_err, &global_err, 1, HYPRE_MPI_INT, hypre_MPI_LOR, comm);
      if (global_err)
      {
         hypre_FSAIFreeBitFSAI(fsai_vdata);
         if (!hypre_error_flag)
         {
            hypre_error(HYPRE_ERROR_GENERIC);
         }
         HYPRE_ANNOTATE_FUNC_END;
         return hypre_error_flag;
      }
   }
   else
   {
      z_work = hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);
      hypre_ParVectorInitialize(z_work);
      hypre_ParFSAIDataZWork(fsai_data) = z_work;

      /* Create the matrix G */
      if (algo_type == 1 || algo_type == 2)
      {
         max_nnzrow_diag_G = max_steps * max_step_size + 1;
      }
      else
      {
         max_nnzrow_diag_G = max_nnz_row + 1;
      }
      max_nonzeros_diag_G = num_rows_diag_A * max_nnzrow_diag_G;
      G = hypre_ParCSRMatrixCreate(comm, num_rows_A, num_cols_A,
                                   row_starts_A, col_starts_A,
                                   0, max_nonzeros_diag_G, 0);
      hypre_ParFSAIDataGmat(fsai_data) = G;

      /* Initialize and compute lower triangular factor G */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      HYPRE_MemoryLocation  memloc_A = hypre_ParCSRMatrixMemoryLocation(A);
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memloc_A);

      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_FSAISetupDevice(fsai_vdata, A, f, u);
      }
      else
#endif
      {
         /* Initialize matrix */
         hypre_ParCSRMatrixInitialize(G);

         switch (algo_type)
         {
            case 1:
               // TODO: Change name to hypre_FSAISetupAdaptive
               hypre_FSAISetupNative(fsai_vdata, A, f, u);
               break;

            case 2:
               // TODO: Change name to hypre_FSAISetupAdaptiveOMPDynamic
               hypre_FSAISetupOMPDyn(fsai_vdata, A, f, u);
               break;

            default:
               hypre_FSAISetupNative(fsai_vdata, A, f, u);
               break;
         }
      }

      /* Compute G^T */
      G  = hypre_ParFSAIDataGmat(fsai_data);
      hypre_ParCSRMatrixTranspose(G, &hypre_ParFSAIDataGTmat(fsai_data), 1);
   }

   /* Update omega if requested */
   if (eig_max_iters)
   {
      hypre_FSAIComputeOmega(fsai_vdata, A);
   }

   /* Print setup info */
   if (print_level == 1)
   {
      hypre_FSAIPrintStats(fsai_data, A);
   }
   else if (print_level > 2 && hypre_ParFSAIDataGmat(fsai_data))
   {
      char filename[] = "FSAI.out.G.ij";
      hypre_ParCSRMatrixPrintIJ(hypre_ParFSAIDataGmat(fsai_data), 0, 0, filename);
   }

#if defined (DEBUG_FSAI)
#if !defined (HYPRE_USING_GPU) ||
   (defined (HYPRE_USING_GPU) && defined (HYPRE_USING_UNIFIED_MEMORY))
   hypre_FSAIDumpLocalLSDense(fsai_vdata, "fsai_dense_ls.out", A);
#endif
#endif

   HYPRE_ANNOTATE_FUNC_END;

   if (!hypre_error_flag)
   {
      hypre_SolverSetIsSetup((hypre_Solver *) fsai_vdata);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIPrintStats
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIPrintStats( void *fsai_vdata,
                      hypre_ParCSRMatrix *A )
{
   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int               algo_type        = hypre_ParFSAIDataAlgoType(fsai_data);
   HYPRE_Int               local_solve_type = hypre_ParFSAIDataLocalSolveType(fsai_data);
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               max_nnz_row      = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   HYPRE_Int               num_levels       = hypre_ParFSAIDataNumLevels(fsai_data);
   HYPRE_Real              threshold        = hypre_ParFSAIDataThreshold(fsai_data);
   HYPRE_Int               eig_max_iters    = hypre_ParFSAIDataEigMaxIters(fsai_data);
   HYPRE_Real              density;

   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);

   /* Local variables */
   HYPRE_Int               nprocs;
   HYPRE_Int               my_id;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &nprocs);
   hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);

   /* Compute density */
   if (G)
   {
      hypre_ParCSRMatrixSetDNumNonzeros(G);
      hypre_ParCSRMatrixSetDNumNonzeros(A);
      density = (HYPRE_Real)(hypre_ParCSRMatrixDNumNonzeros(G) /
                             hypre_ParCSRMatrixDNumNonzeros(A));
   }
   else
   {
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      double local_g_nnz = 0.5 * ((double) hypre_CSRMatrixNumNonzeros(A_diag) +
                                  (double) hypre_CSRMatrixNumRows(A_diag));
      double global_g_nnz = 0.0;
      hypre_MPI_Allreduce(&local_g_nnz, &global_g_nnz, 1, hypre_MPI_DOUBLE,
                          hypre_MPI_SUM, hypre_ParCSRMatrixComm(A));
      hypre_ParCSRMatrixSetDNumNonzeros(A);
      density = (hypre_ParCSRMatrixDNumNonzeros(A) > 0.0)
                ? (HYPRE_Real)(global_g_nnz / hypre_ParCSRMatrixDNumNonzeros(A))
                : 0.0;
   }
   hypre_ParFSAIDataDensity(fsai_data) = density;

   if (!my_id)
   {
      hypre_printf("*************************\n");
      hypre_printf("* HYPRE FSAI Setup Info *\n");
      hypre_printf("*************************\n\n");

      hypre_printf("+---------------------------+\n");
      hypre_printf("| No. MPI tasks:     %6d |\n", nprocs);
      hypre_printf("| No. threads:       %6d |\n", hypre_NumThreads());
      hypre_printf("| Algorithm type:    %6d |\n", algo_type);
      hypre_printf("| Local solve type:  %6d |\n", local_solve_type);
      if (algo_type == 1 || algo_type == 2)
      {
         hypre_printf("| Max no. steps:     %6d |\n", max_steps);
         hypre_printf("| Max step size:     %6d |\n", max_step_size);
         hypre_printf("| Kap grad tol:    %8.1e |\n", kap_tolerance);
      }
      else
      {
         hypre_printf("| Max nnz. row:      %6d |\n", max_nnz_row);
         hypre_printf("| Number of levels:  %6d |\n", num_levels);
         hypre_printf("| Threshold:       %8.1e |\n", threshold);
      }
      hypre_printf("| Prec. density:   %8.3f |\n", density);
      hypre_printf("| Eig max iters:     %6d |\n", eig_max_iters);
      hypre_printf("| Omega factor:    %8.3f |\n", hypre_ParFSAIDataOmega(fsai_data));
      hypre_printf("+---------------------------+\n");

      hypre_printf("\n\n");
   }

   return hypre_error_flag;
}

/*****************************************************************************
 * hypre_FSAIComputeOmega
 *
 * Approximates the relaxation factor omega with 1/eigmax(G^T*G*A), where the
 * maximum eigenvalue is computed with a fixed number of iterations via the
 * power method.
 ******************************************************************************/

HYPRE_Int
hypre_FSAIComputeOmega( void               *fsai_vdata,
                        hypre_ParCSRMatrix *A )
{
   hypre_ParFSAIData    *fsai_data       = (hypre_ParFSAIData*) fsai_vdata;
   hypre_ParCSRMatrix   *G               = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix   *GT              = hypre_ParFSAIDataGTmat(fsai_data);
   hypre_ParVector      *r_work          = hypre_ParFSAIDataRWork(fsai_data);
   hypre_ParVector      *z_work          = hypre_ParFSAIDataZWork(fsai_data);
   HYPRE_Int             eig_max_iters   = hypre_ParFSAIDataEigMaxIters(fsai_data);
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_ParVector      *eigvec;
   hypre_ParVector      *eigvec_old;

   HYPRE_Int             i;
   HYPRE_Real            norm, invnorm, lambda, omega;

   eigvec_old = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
   eigvec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                  hypre_ParCSRMatrixGlobalNumRows(A),
                                  hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(eigvec, memory_location);
   hypre_ParVectorInitialize_v2(eigvec_old, memory_location);

#if defined(HYPRE_USING_GPU)
   /* Make random number generation faster on GPUs */
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      hypre_Vector  *eigvec_local = hypre_ParVectorLocalVector(eigvec);
      HYPRE_Complex *eigvec_data  = hypre_VectorData(eigvec_local);
      HYPRE_Int      eigvec_size  = hypre_VectorSize(eigvec_local);

      hypre_CurandUniform(eigvec_size, eigvec_data, 0, 0, 0, 0);
   }
   else
#endif
   {
      hypre_ParVectorSetRandomValues(eigvec, 256);
   }

   /* Power method iteration */
   for (i = 0; i < eig_max_iters; i++)
   {
      norm = hypre_ParVectorInnerProd(eigvec, eigvec);
      invnorm = 1.0 / hypre_sqrt(norm);
      hypre_ParVectorScale(invnorm, eigvec);

      if (i == (eig_max_iters - 1))
      {
         hypre_ParVectorCopy(eigvec, eigvec_old);
      }

      /* eigvec = GT * G * A * eigvec */
      hypre_ParCSRMatrixMatvec(1.0, A, eigvec, 0.0, r_work);
      if (hypre_ParFSAIDataAlgoType(fsai_data) == 4)
      {
         HYPRE_Real saved_omega = hypre_ParFSAIDataOmega(fsai_data);
         hypre_ParFSAIDataOmega(fsai_data) = 1.0;
         hypre_FSAIApply(fsai_vdata, 0.0, r_work, eigvec);
         hypre_ParFSAIDataOmega(fsai_data) = saved_omega;
      }
      else
      {
         hypre_ParCSRMatrixMatvec(1.0, G,  r_work, 0.0, z_work);
         hypre_ParCSRMatrixMatvec(1.0, GT, z_work, 0.0, eigvec);
      }
   }
   norm = hypre_ParVectorInnerProd(eigvec, eigvec_old);
   lambda = hypre_sqrt(norm);

   /* Check lambda */
   if (lambda < HYPRE_REAL_EPSILON)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Found small lambda. Reseting it to one!");
      lambda = 1.0;
   }

   /* Free memory */
   hypre_ParVectorDestroy(eigvec_old);
   hypre_ParVectorDestroy(eigvec);

   /* Update omega */
   omega = 1.0 / lambda;
   hypre_FSAISetOmega(fsai_vdata, omega);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIDumpLocalLSDense
 *
 * Dump local linear systems to file. Matrices are written in dense format.
 * This functions serves for debugging.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIDumpLocalLSDense( void               *fsai_vdata,
                            const char         *filename,
                            hypre_ParCSRMatrix *A )
{
   hypre_ParFSAIData      *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int               max_steps = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);
   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag = hypre_ParCSRMatrixDiag(G);
   HYPRE_Int              *G_i = hypre_CSRMatrixI(G_diag);
   HYPRE_Int              *G_j = hypre_CSRMatrixJ(G_diag);
   HYPRE_Int               num_rows_diag_G = hypre_CSRMatrixNumRows(G_diag);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Int              *A_j              = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex          *A_a              = hypre_CSRMatrixData(A_diag);

   FILE                   *fp;
   char                    new_filename[1024];
   HYPRE_Int               myid;
   HYPRE_Int               i, j, k, m, n;
   HYPRE_Int               ii, jj;
   HYPRE_Int               nnz, col, index;
   HYPRE_Int              *indices;
   HYPRE_Int              *marker;
   HYPRE_Real             *data;
   HYPRE_Int               data_size;
   HYPRE_Real              density;
   HYPRE_Int               width = 20; //6
   HYPRE_Int               prec  = 16; //2

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((fp = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   /* Allocate memory */
   data_size = (max_steps * max_step_size) *
               (max_steps * max_step_size + 1);
   indices = hypre_CTAlloc(HYPRE_Int, data_size, HYPRE_MEMORY_HOST);
   data    = hypre_CTAlloc(HYPRE_Real, data_size, HYPRE_MEMORY_HOST);
   marker  = hypre_TAlloc(HYPRE_Int, num_rows_diag_G, HYPRE_MEMORY_HOST);
   hypre_Memset(marker, -1, (size_t) num_rows_diag_G * sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

   /* Write header info */
   hypre_fprintf(fp, "num_linear_sys = %d\n", num_rows_diag_G);
   hypre_fprintf(fp, "max_data_size = %d\n", data_size);
   hypre_fprintf(fp, "max_num_steps = %d\n", hypre_ParFSAIDataMaxSteps(fsai_data));
   hypre_fprintf(fp, "max_step_size = %d\n", hypre_ParFSAIDataMaxStepSize(fsai_data));
   hypre_fprintf(fp, "max_step_size = %g\n", hypre_ParFSAIDataKapTolerance(fsai_data));
   hypre_fprintf(fp, "algo_type = %d\n\n", hypre_ParFSAIDataAlgoType(fsai_data));

   /* Write local full linear systems */
   for (i = 0; i < num_rows_diag_G; i++)
   {
      /* Build marker array */
      n = G_i[i + 1] - G_i[i] - 1;
      m = n + 1;
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         marker[G_j[j]] = j - G_i[i] - 1;
      }

      /* Gather matrix coefficients */
      nnz = 0;
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         for (k = A_i[G_j[j]]; k < A_i[G_j[j] + 1]; k++)
         {
            if ((col = marker[A_j[k]]) >= 0)
            {
               /* Add A(i,j) entry */
               index = (j - G_i[i] - 1) * n + col;
               data[index] = A_a[k];
               indices[nnz] = index;
               nnz++;
            }
         }
      }
      density = (n > 0) ? (HYPRE_Real) nnz / (n * n) : 0.0;

      /* Gather RHS coefficients */
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         if ((col = marker[A_j[j]]) >= 0)
         {
            index = (m - 1) * n + col;
            data[index] = A_a[j];
            indices[nnz] = index;
            nnz++;
         }
      }

      /* Write coefficients to file */
      hypre_fprintf(fp, "id = %d, (m, n) = (%d, %d), rho = %.3f\n", i, m, n, density);
      for (ii = 0; ii < n; ii++)
      {
         for (jj = 0; jj < n; jj++)
         {
            hypre_fprintf(fp, "%*.*f ", width, prec, data[ii * n + jj]);
         }
         hypre_fprintf(fp, "\n");
      }
      for (jj = 0; jj < n; jj++)
      {
         hypre_fprintf(fp, "%*.*f ", width, prec, data[ii * n + jj]);
      }
      hypre_fprintf(fp, "\n");


      /* Reset work arrays */
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         marker[G_j[j]] = -1;
      }

      for (k = 0; k < nnz; k++)
      {
         data[indices[k]] = 0.0;
      }
   }

   /* Close stream */
   fclose(fp);

   /* Free memory */
   hypre_TFree(indices, HYPRE_MEMORY_HOST);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   hypre_TFree(data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
