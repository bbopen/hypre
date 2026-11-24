/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Unit tests for 4-bit zero-index BitFSAI (algo_type = 4) in ParFSAI.
 *--------------------------------------------------------------------------*/

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.h"
#include "hypre_unit.h"

#if !defined(HYPRE_COMPLEX)

static hypre_ParCSRMatrix *
CreateSelfParCSR(HYPRE_Int         n,
                 HYPRE_Int         nnz,
                 const HYPRE_Int  *rp,
                 const HYPRE_Int  *ci,
                 const HYPRE_Real *val)
{
   HYPRE_BigInt        starts[2] = {0, (HYPRE_BigInt) n};
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag;
   HYPRE_Int           i;

   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_SELF, (HYPRE_BigInt) n, (HYPRE_BigInt) n,
                                starts, starts, 0, nnz, 0);
   hypre_ParCSRMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   diag = hypre_ParCSRMatrixDiag(A);
   for (i = 0; i <= n; i++)
   {
      hypre_CSRMatrixI(diag)[i] = rp[i];
   }
   for (i = 0; i < nnz; i++)
   {
      hypre_CSRMatrixJ(diag)[i]    = ci[i];
      hypre_CSRMatrixData(diag)[i] = (HYPRE_Complex) val[i];
   }
   return A;
}

static hypre_ParVector *
CreateSelfParVector(HYPRE_Int n, const HYPRE_Real *vals)
{
   HYPRE_BigInt     starts[2] = {0, (HYPRE_BigInt) n};
   hypre_ParVector *v;
   HYPRE_Complex   *data;
   HYPRE_Int        i;

   v = hypre_ParVectorCreate(hypre_MPI_COMM_SELF, (HYPRE_BigInt) n, starts);
   hypre_ParVectorInitialize_v2(v, HYPRE_MEMORY_HOST);
   data = hypre_VectorData(hypre_ParVectorLocalVector(v));
   for (i = 0; i < n; i++)
   {
      data[i] = vals ? (HYPRE_Complex) vals[i] : (HYPRE_Complex) 0.0;
   }
   return v;
}

static HYPRE_Int
TestDenseOracleAndEnergy(void)
{
   const HYPRE_Int nx = 4, ny = 4, n = nx * ny;
   HYPRE_Int      *rp = hypre_CTAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int      *ci = hypre_CTAlloc(HYPRE_Int, 5 * n, HYPRE_MEMORY_HOST);
   HYPRE_Real     *val = hypre_CTAlloc(HYPRE_Real, 5 * n, HYPRE_MEMORY_HOST);
   HYPRE_Real     *bvals = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   double         *G_dense = hypre_CTAlloc(double, n * n, HYPRE_MEMORY_HOST);
   double         *A_dense = hypre_CTAlloc(double, n * n, HYPRE_MEMORY_HOST);
   double         *t_ref = hypre_CTAlloc(double, n, HYPRE_MEMORY_HOST);
   double         *z_ref = hypre_CTAlloc(double, n, HYPRE_MEMORY_HOST);
   HYPRE_Int       nnz = 0, i, j, k;
   HYPRE_Int       failed = 0;

   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {
         HYPRE_Int row = j * nx + i;
         rp[row] = nnz;
         /* HYPRE convention: diagonal first, unsorted off-diagonals */
         ci[nnz] = row; val[nnz++] = 22.0; A_dense[row * n + row] = 22.0;
         if (i > 0)      { ci[nnz] = row - 1;  val[nnz++] = -1.0;  A_dense[row * n + row - 1]  = -1.0; }
         if (i + 1 < nx) { ci[nnz] = row + 1;  val[nnz++] = -1.0;  A_dense[row * n + row + 1]  = -1.0; }
         if (j > 0)      { ci[nnz] = row - nx; val[nnz++] = -10.0; A_dense[row * n + row - nx] = -10.0; }
         if (j + 1 < ny) { ci[nnz] = row + nx; val[nnz++] = -10.0; A_dense[row * n + row + nx] = -10.0; }
         bvals[row] = (HYPRE_Real) sin(0.3 * (row + 1));
      }
   }
   rp[n] = nnz;

   hypre_ParCSRMatrix *A = CreateSelfParCSR(n, nnz, rp, ci, val);
   hypre_ParVector    *b = CreateSelfParVector(n, bvals);
   hypre_ParVector    *x = CreateSelfParVector(n, NULL);
   HYPRE_Solver        solver;

   HYPRE_FSAICreate(&solver);
   HYPRE_FSAISetAlgoType(solver, 4);
   HYPRE_FSAISetOmega(solver, 0.72);
   HYPRE_FSAISetMaxIterations(solver, 1);
   HYPRE_FSAISetZeroGuess(solver, 1);
   HYPRE_FSAISetEigMaxIters(solver, 0);

   HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
   HYPRE_FSAISolve(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);

   hypre_ParFSAIData   *fsai_data = (hypre_ParFSAIData *) solver;
   const float         *g_diag    = hypre_ParFSAIDataBitFSAIGDiag(fsai_data);
   const float         *g_scale   = hypre_ParFSAIDataBitFSAIGScale(fsai_data);
   const unsigned char *q_packed  = hypre_ParFSAIDataBitFSAIQPacked(fsai_data);

   /* Reconstruct dense G_q from stored BitFSAI factor and check stored A-energy + G_q^T G_q b */
   for (i = 0; i < n; i++)
   {
      G_dense[i * n + i] = (double) g_diag[i];
      for (k = rp[i]; k < rp[i + 1]; k++)
      {
         HYPRE_Int col = ci[k];
         if (col < i)
         {
            unsigned char byte = q_packed[k >> 1];
            unsigned char nib  = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            double w = (double)(((float)((int) nib - 8)) * (1.0f / 7.0f));
            G_dense[i * n + col] = (double) g_scale[i] * w;
         }
      }
   }

   for (i = 0; i < n; i++)
   {
      double energy = 0.0;
      for (j = 0; j < n; j++)
      {
         double aj = 0.0;
         for (k = 0; k < n; k++)
         {
            aj += A_dense[j * n + k] * G_dense[i * n + k];
         }
         energy += G_dense[i * n + j] * aj;
      }
      if (!isfinite(energy) || fabs(energy - 1.0) > 1.0e-4)
      {
         hypre_printf("FAIL: row %d stored A-energy = %.8f\n", (int) i, energy);
         failed = 1;
      }
   }

   for (i = 0; i < n; i++)
   {
      double s = 0.0;
      for (j = 0; j < n; j++) { s += G_dense[i * n + j] * (double) bvals[j]; }
      t_ref[i] = s;
   }
   HYPRE_Complex *xdata = hypre_VectorData(hypre_ParVectorLocalVector(x));
   for (j = 0; j < n; j++)
   {
      double s = 0.0;
      for (i = 0; i < n; i++) { s += G_dense[i * n + j] * t_ref[i]; }
      z_ref[j] = 0.72 * s;
      if (!isfinite((double) xdata[j]) || fabs((double) xdata[j] - z_ref[j]) > 1.0e-5)
      {
         hypre_printf("FAIL: dense oracle mismatch at %d: got %.12e expected %.12e\n",
                      (int) j, (double) xdata[j], z_ref[j]);
         failed = 1;
      }
   }

   /* Re-setup: same-type (4->4) and cross-type (4->1->4) must succeed cleanly */
   HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
   HYPRE_FSAISetAlgoType(solver, 1);
   HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
   HYPRE_FSAISetAlgoType(solver, 4);
   HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
   if (HYPRE_GetError() != 0)
   {
      hypre_printf("FAIL: re-setup raised error %d\n", (int) HYPRE_GetError());
      failed = 1;
   }

   HYPRE_FSAIDestroy(solver);
   hypre_ParVectorDestroy(b);
   hypre_ParVectorDestroy(x);
   hypre_ParCSRMatrixDestroy(A);
   hypre_TFree(rp, HYPRE_MEMORY_HOST);
   hypre_TFree(ci, HYPRE_MEMORY_HOST);
   hypre_TFree(val, HYPRE_MEMORY_HOST);
   hypre_TFree(bvals, HYPRE_MEMORY_HOST);
   hypre_TFree(G_dense, HYPRE_MEMORY_HOST);
   hypre_TFree(A_dense, HYPRE_MEMORY_HOST);
   hypre_TFree(t_ref, HYPRE_MEMORY_HOST);
   hypre_TFree(z_ref, HYPRE_MEMORY_HOST);
   return failed;
}

static HYPRE_Int
TestNumericalEdgeCases(void)
{
   HYPRE_Int failed = 0;

   /* 1. Single-precision wide damping no-overflow: A=[2^-130], b=[1], omega=1/8 -> 2^127 */
   {
      HYPRE_Int  rp[2] = {0, 1};
      HYPRE_Int  ci[1] = {0};
      HYPRE_Real val[1] = {(HYPRE_Real) ldexp(1.0, -130)};
      HYPRE_Real bv[1]  = {(HYPRE_Real) 1.0};
      hypre_ParCSRMatrix *A = CreateSelfParCSR(1, 1, rp, ci, val);
      hypre_ParVector    *b = CreateSelfParVector(1, bv);
      hypre_ParVector    *x = CreateSelfParVector(1, NULL);
      HYPRE_Solver        solver;

      if (val[0] > 0.0)
      {
         HYPRE_FSAICreate(&solver);
         HYPRE_FSAISetAlgoType(solver, 4);
         HYPRE_FSAISetOmega(solver, 0.125);
         HYPRE_FSAISetMaxIterations(solver, 1);
         HYPRE_FSAISetZeroGuess(solver, 1);
         HYPRE_FSAISetEigMaxIters(solver, 0);
         HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
         HYPRE_FSAISolve(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
         double got = (double) hypre_VectorData(hypre_ParVectorLocalVector(x))[0];
         double exp = ldexp(1.0, 127);
         if (!isfinite(got) || fabs(got - exp) / exp > 1.0e-5)
         {
            hypre_printf("FAIL: wide damping overflow test got=%.6e exp=%.6e\n", got, exp);
            failed = 1;
         }
         HYPRE_FSAIDestroy(solver);
      }
      hypre_ParVectorDestroy(b);
      hypre_ParVectorDestroy(x);
      hypre_ParCSRMatrixDestroy(A);
   }

   /* 2. Ill-conditioned k=7000 triggers Jacobi fallback */
   {
      double     k = 7000.0;
      HYPRE_Int  rp[3] = {0, 2, 4};
      HYPRE_Int  ci[4] = {0, 1, 1, 0};
      HYPRE_Real val[4] = {
         (HYPRE_Real)(k * k + 1.0), (HYPRE_Real)(k * (k + 1.0) + 1.0),
         (HYPRE_Real)((k + 1.0) * (k + 1.0) + 1.0), (HYPRE_Real)(k * (k + 1.0) + 1.0)
      };
      HYPRE_Real bv[2] = {1.0, 1.0};
      hypre_ParCSRMatrix *A = CreateSelfParCSR(2, 4, rp, ci, val);
      hypre_ParVector    *b = CreateSelfParVector(2, bv);
      hypre_ParVector    *x = CreateSelfParVector(2, NULL);
      HYPRE_Solver        solver;

      HYPRE_FSAICreate(&solver);
      HYPRE_FSAISetAlgoType(solver, 4);
      HYPRE_FSAISetOmega(solver, 1.0);
      HYPRE_FSAISetMaxIterations(solver, 1);
      HYPRE_FSAISetZeroGuess(solver, 1);
      HYPRE_FSAISetEigMaxIters(solver, 0);
      HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
      HYPRE_FSAISolve(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
      HYPRE_Complex *xd = hypre_VectorData(hypre_ParVectorLocalVector(x));
      double e0 = 1.0 / (k * k + 1.0);
      double e1 = 1.0 / ((k + 1.0) * (k + 1.0) + 1.0);
      if (!isfinite((double) xd[0]) || !isfinite((double) xd[1]) ||
          fabs((double) xd[0] - e0) / e0 > 1.0e-4 ||
          fabs((double) xd[1] - e1) / e1 > 1.0e-4)
      {
         hypre_printf("FAIL: k=7000 fallback got=(%.6e,%.6e) exp=(%.6e,%.6e)\n",
                      (double) xd[0], (double) xd[1], e0, e1);
         failed = 1;
      }
      HYPRE_FSAIDestroy(solver);
      hypre_ParVectorDestroy(b);
      hypre_ParVectorDestroy(x);
      hypre_ParCSRMatrixDestroy(A);
   }

   /* 3a. Nonsymmetric pattern (lower-only, missing upper mirror) must fail cleanly */
   {
      HYPRE_Int  rp[3] = {0, 1, 3};
      HYPRE_Int  ci[3] = {0, 1, 0};
      HYPRE_Real val[3] = {4.0, 4.0, -1.0};
      HYPRE_Real bv[2] = {1.0, 1.0};
      hypre_ParCSRMatrix *A = CreateSelfParCSR(2, 3, rp, ci, val);
      hypre_ParVector    *b = CreateSelfParVector(2, bv);
      hypre_ParVector    *x = CreateSelfParVector(2, NULL);
      HYPRE_Solver        solver;

      HYPRE_ClearAllErrors();
      HYPRE_FSAICreate(&solver);
      HYPRE_FSAISetAlgoType(solver, 4);
      HYPRE_FSAISetEigMaxIters(solver, 0);
      HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
      if (HYPRE_GetError() == 0)
      {
         hypre_printf("FAIL: lower-only nonsymmetric matrix did not raise error\n");
         failed = 1;
      }
      HYPRE_ClearAllErrors();
      HYPRE_FSAIDestroy(solver);
      hypre_ParVectorDestroy(b);
      hypre_ParVectorDestroy(x);
      hypre_ParCSRMatrixDestroy(A);
   }

   /* 3b. Nonsymmetric pattern (upper-only, missing lower mirror) must fail cleanly */
   {
      HYPRE_Int  rp[3] = {0, 2, 3};
      HYPRE_Int  ci[3] = {0, 1, 1};
      HYPRE_Real val[3] = {4.0, -1.0, 4.0};
      HYPRE_Real bv[2] = {1.0, 1.0};
      hypre_ParCSRMatrix *A = CreateSelfParCSR(2, 3, rp, ci, val);
      hypre_ParVector    *b = CreateSelfParVector(2, bv);
      hypre_ParVector    *x = CreateSelfParVector(2, NULL);
      HYPRE_Solver        solver;

      HYPRE_ClearAllErrors();
      HYPRE_FSAICreate(&solver);
      HYPRE_FSAISetAlgoType(solver, 4);
      HYPRE_FSAISetEigMaxIters(solver, 0);
      HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
      if (HYPRE_GetError() == 0)
      {
         hypre_printf("FAIL: upper-only nonsymmetric matrix did not raise error\n");
         failed = 1;
      }
      HYPRE_ClearAllErrors();
      HYPRE_FSAIDestroy(solver);
      hypre_ParVectorDestroy(b);
      hypre_ParVectorDestroy(x);
      hypre_ParCSRMatrixDestroy(A);
   }

#if !defined(HYPRE_SINGLE)
   /* 4. Unrepresentable FP32 diagonal 1e100 must fail cleanly */
   {
      HYPRE_Int  rp[3] = {0, 1, 2};
      HYPRE_Int  ci[2] = {0, 1};
      HYPRE_Real val[2] = {1.0e100, 1.0e100};
      HYPRE_Real bv[2] = {1.0, 1.0};
      hypre_ParCSRMatrix *A = CreateSelfParCSR(2, 2, rp, ci, val);
      hypre_ParVector    *b = CreateSelfParVector(2, bv);
      hypre_ParVector    *x = CreateSelfParVector(2, NULL);
      HYPRE_Solver        solver;

      HYPRE_ClearAllErrors();
      HYPRE_FSAICreate(&solver);
      HYPRE_FSAISetAlgoType(solver, 4);
      HYPRE_FSAISetEigMaxIters(solver, 0);
      HYPRE_FSAISetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
      if (HYPRE_GetError() == 0)
      {
         hypre_printf("FAIL: 1e100 unrepresentable diagonal did not raise error\n");
         failed = 1;
      }
      HYPRE_ClearAllErrors();
      HYPRE_FSAIDestroy(solver);
      hypre_ParVectorDestroy(b);
      hypre_ParVectorDestroy(x);
      hypre_ParCSRMatrixDestroy(A);
   }
#endif

   return failed;
}

#endif /* !defined(HYPRE_COMPLEX) */

int
main(int argc, char *argv[])
{
   HYPRE_Int failed = 0;

   hypre_MPI_Init(&argc, &argv);
   HYPRE_Initialize();
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);

#if !defined(HYPRE_COMPLEX)
   {
      HYPRE_Int f1 = TestDenseOracleAndEnergy();
      hypre_printf("Test1 (DenseOracleAndEnergy): %s\n", f1 ? "FAILED" : "PASSED");
      failed |= f1;
   }
   {
      HYPRE_Int f2 = TestNumericalEdgeCases();
      hypre_printf("Test2 (NumericalEdgeCases): %s\n", f2 ? "FAILED" : "PASSED");
      failed |= f2;
   }
#else
   hypre_printf("Test1 (ComplexSkip): PASSED\n");
#endif

   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return failed ? 1 : 0;
}
