/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

/******************************************************************************
 *
 * FSAI solve routine
 *
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------
 * hypre_FSAISolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISolve( void               *fsai_vdata,
                 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *b,
                 hypre_ParVector    *x )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   hypre_ParFSAIData   *fsai_data   = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   hypre_ParVector     *r_work      = hypre_ParFSAIDataRWork(fsai_data);
   HYPRE_Real           tol         = hypre_ParFSAIDataTolerance(fsai_data);
   HYPRE_Int            zero_guess  = hypre_ParFSAIDataZeroGuess(fsai_data);
   HYPRE_Int            max_iter    = hypre_ParFSAIDataMaxIterations(fsai_data);
   HYPRE_Int            print_level = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int            logging     = hypre_ParFSAIDataLogging(fsai_data);

   /* Local variables */
   HYPRE_Int            iter, my_id;
   HYPRE_Real           old_resnorm, resnorm = 0.0, rel_resnorm;
   HYPRE_Complex        one = 1.0;
   HYPRE_Complex        neg_one = -1.0;
   HYPRE_Complex        zero = 0.0;

   /* Sanity check */
   if (hypre_ParVectorNumVectors(b) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "FSAI doesn't support multicomponent vectors");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("FSAISolve");

   hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------
    * Preconditioned Richardson - Main solver loop
    * x(k+1) = x(k) + omega * (G^T*G) * (b - A*x(k))
    * ----------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1)
   {
      hypre_printf("\n\n FSAI SOLVER SOLUTION INFO:\n");
   }

   iter        = 0;
   rel_resnorm = resnorm = 1.0;

   if (my_id == 0 && print_level > 1)
   {
      hypre_printf("                new         relative\n");
      hypre_printf("    iter #      res norm    res norm\n");
      hypre_printf("    --------    --------    --------\n");
   }

   if (max_iter > 0)
   {
      /* First iteration */
      if (zero_guess)
      {
         /* Compute: x(k+1) = omega*G^T*G*b */
         hypre_FSAIApply(fsai_vdata, zero, b, x);
      }
      else
      {
         /* Compute: x(k+1) = x(k) + omega*G^T*G*(b - A*x(k)) */
         hypre_ParCSRMatrixMatvecOutOfPlace(neg_one, A, x, one, b, r_work);
         hypre_FSAIApply(fsai_vdata, one, r_work, x);
      }

      /* Update iteration count */
      iter++;
   }
   else
   {
      hypre_ParVectorCopy(b, x);
   }

   /* Apply remaining iterations */
   for (; iter < max_iter; iter++)
   {
      /* Update residual */
      hypre_ParCSRMatrixMatvecOutOfPlace(neg_one, A, x, one, b, r_work);

      if (tol > 0.0)
      {
         old_resnorm = resnorm;
         resnorm = hypre_ParVectorInnerProd(r_work, r_work);

         /* Compute rel_resnorm */
         rel_resnorm = resnorm / old_resnorm;

         if (my_id == 0 && print_level > 1)
         {
            hypre_printf("    %e          %e          %e\n", iter, resnorm, rel_resnorm);
         }

         /* Exit if convergence tolerance has been achieved */
         if (rel_resnorm >= tol)
         {
            break;
         }
      }

      /* Compute: x(k+1) = x(k) + omega*inv(M)*r */
      hypre_FSAIApply(fsai_vdata, one, r_work, x);
   }

   if (logging > 1)
   {
      hypre_ParFSAIDataNumIterations(fsai_data) = iter;
      hypre_ParFSAIDataRelResNorm(fsai_data)    = rel_resnorm;
   }
   else
   {
      hypre_ParFSAIDataNumIterations(fsai_data) = 0;
      hypre_ParFSAIDataRelResNorm(fsai_data)    = 0.0;
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_FSAIApplyBitFSAIHost
 *
 * Computes x = alpha * x + omega * G_q^T * G_q * b on host using
 * zero-duplicate-index 4-bit mirrored CSR weights and a 4 B/row
 * scale-factored intermediate buffer y_i = s_i * (G_q * b)_i.
 *--------------------------------------------------------------------*/

static HYPRE_Int
hypre_FSAIApplyBitFSAIHost( void            *fsai_vdata,
                            HYPRE_Complex    alpha,
                            hypre_ParVector *b,
                            hypre_ParVector *x )
{
#if defined(HYPRE_COMPLEX)
   HYPRE_UNUSED_VAR(fsai_vdata);
   HYPRE_UNUSED_VAR(alpha);
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "BitFSAI (algo_type = 4) does not support complex arithmetic\n");
   return hypre_error_flag;
#else
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
   const HYPRE_Int     *A_i;
   const HYPRE_Int     *A_j;
   const HYPRE_Complex *b_data;
   HYPRE_Complex       *x_data;
   HYPRE_Int            i, j;

   if (num_rows <= 0 || !A_diag)
   {
      return hypre_error_flag;
   }

   A_i    = hypre_CSRMatrixI(A_diag);
   A_j    = hypre_CSRMatrixJ(A_diag);
   b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   if (wide_apply)
   {
      /* Wide Pass 1: unscaled FP64 t_i = (G_q * b)_i */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
      for (i = 0; i < num_rows; i++)
      {
         HYPRE_Int k;
         double    sc  = (double) g_scale[i];
         double    acc = (double) g_diag[i] * (double) b_data[i];

         for (k = A_i[i]; k < A_i[i + 1]; k++)
         {
            HYPRE_Int col = A_j[k];
            if (col < i)
            {
               unsigned char byte = q_packed[k >> 1];
               unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
               double        w    = (double)(((float)((int) nib - 8)) * (1.0f / 7.0f));
               acc += (sc * w) * (double) b_data[col];
            }
         }
         t_work[i] = acc;
      }

      /* Wide Pass 2: x_j = alpha * x_j + omega * (G_q^T * t)_j */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for schedule(static) private(j)
#endif
      for (j = 0; j < num_rows; j++)
      {
         HYPRE_Int k;
         double    acc = (double) g_diag[j] * t_work[j];

         for (k = A_i[j]; k < A_i[j + 1]; k++)
         {
            HYPRE_Int col = A_j[k];
            if (col > j)
            {
               unsigned char byte = q_packed[k >> 1];
               unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
               double        w    = (double)(((float)((int) nib - 8)) * (1.0f / 7.0f));
               acc += ((double) g_scale[col] * w) * t_work[col];
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
   else
   {
      /* Fast FP32 Pass 1: y_i = s_i * (G_q * b)_i (4 B/row) */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
      for (i = 0; i < num_rows; i++)
      {
         HYPRE_Int k;
         float     sc  = g_scale[i];
         float     acc = 0.0f;
         float     ti;

         for (k = A_i[i]; k < A_i[i + 1]; k++)
         {
            HYPRE_Int col = A_j[k];
            if (col < i)
            {
               unsigned char byte = q_packed[k >> 1];
               unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
               float         w    = ((float)((int) nib - 8)) * (1.0f / 7.0f);
               acc += w * (float) b_data[col];
            }
         }
         ti        = sc * acc + g_diag[i] * (float) b_data[i];
         y_work[i] = sc * ti;
      }

      /* Fast FP32 Pass 2: x_j = alpha * x_j + omega * (G_q^T * t)_j */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for schedule(static) private(j)
#endif
      for (j = 0; j < num_rows; j++)
      {
         HYPRE_Int k;
         float     sc_j = g_scale[j];
         float     acc  = (sc_j != 0.0f) ? (g_diag[j] * (y_work[j] / sc_j)) : 0.0f;

         for (k = A_i[j]; k < A_i[j + 1]; k++)
         {
            HYPRE_Int col = A_j[k];
            if (col > j)
            {
               unsigned char byte = q_packed[k >> 1];
               unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
               float         w    = ((float)((int) nib - 8)) * (1.0f / 7.0f);
               acc += w * y_work[col];
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

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------
 * hypre_FSAIApply
 *
 * Computes x(k+1) = alpha*x(k) + omega*G^T*G*b
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIApply( void               *fsai_vdata,
                 HYPRE_Complex       alpha,
                 hypre_ParVector    *b,
                 hypre_ParVector    *x )
{
   hypre_ParFSAIData   *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   HYPRE_Int            algo_type = hypre_ParFSAIDataAlgoType(fsai_data);
   hypre_ParCSRMatrix  *G         = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix  *GT        = hypre_ParFSAIDataGTmat(fsai_data);
   hypre_ParVector     *z_work    = hypre_ParFSAIDataZWork(fsai_data);
   HYPRE_Real           omega     = hypre_ParFSAIDataOmega(fsai_data);

   HYPRE_Complex        one  = 1.0;
   HYPRE_Complex        zero = 0.0;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("FSAIApply");

   if (algo_type == 4)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      HYPRE_MemoryLocation memloc = hypre_ParFSAIDataBitFSAIMemLoc(fsai_data);
      if (hypre_GetExecPolicy1(memloc) == HYPRE_EXEC_DEVICE)
      {
         hypre_FSAIApplyBitFSAIDevice(fsai_vdata, alpha, b, x);
      }
      else
#elif defined(HYPRE_USING_GPU)
      HYPRE_MemoryLocation memloc = hypre_ParFSAIDataBitFSAIMemLoc(fsai_data);
      if (hypre_GetExecPolicy1(memloc) == HYPRE_EXEC_DEVICE)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "BitFSAI (algo_type = 4) device execution requires CUDA or HIP\n");
      }
      else
#endif
      {
         hypre_FSAIApplyBitFSAIHost(fsai_vdata, alpha, b, x);
      }
   }
   else
   {
      hypre_ParCSRMatrixMatvec(one, G, b, zero, z_work);
      hypre_ParCSRMatrixMatvec(omega, GT, z_work, alpha, x);
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
