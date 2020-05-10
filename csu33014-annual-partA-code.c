//
// CSU33014 Summer 2020 Additional Assignment
// Part A of a two-part assignment
//

// Please examine version each of the following routines with names
// starting partA. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics. Where is cannot be vectorized...

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".

//printf("\nvector[%u-%u]: %f %f %f %f\n", i, i+3, vector[0], vector[1], vector[2], vector[3]);
#include <immintrin.h>
#include <stdio.h>

#include "csu33014-annual-partA-code.h"

#define VECTOR_LEN 4

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void partA_routine0(float * restrict a, float * restrict b,
		    float * restrict c) {
  for (int i = 0; i < 1024; i++ ) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void partA_vectorized0(float * restrict a, float * restrict b,
		    float * restrict c) {
  __m128 a4, b4, c4;
  
  for (int i = 0; i < 1024; i = i+4 ) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float partA_routine1(float * restrict a, float * restrict b,
		     int size) {
  float sum = 0.0;
  for ( int i = 0; i < size; i++ ) {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float partA_vectorized1(float * restrict a, float * restrict b,
		     int size) {
  float sum= 0.0;
  __m128 vSum=  _mm_setzero_ps();
  __m128 vTemp=  _mm_setzero_ps();
  __m128 vTemp2=  _mm_setzero_ps();
  int vecOpCount= size - (size % VECTOR_LEN); //number of vector ops to be performed
  int i= 0;
  for (; i < vecOpCount; i+= VECTOR_LEN) //VECTOR_LEN= 4
  {
    vTemp= _mm_load_ps (&a[i]);
    vTemp2= _mm_load_ps (&b[i]);
    vTemp= _mm_mul_ps (vTemp2, vTemp);
    __m128 shuf= _mm_movehdup_ps(vTemp);        // broadcast elements 3,1 to 2,0
    __m128 sums= _mm_add_ps(vTemp, shuf);
    shuf= _mm_movehl_ps(shuf, sums); // high half -> low half
    sums= _mm_add_ss(sums, shuf);
    sum+= _mm_cvtss_f32(sums);
  }
  for (; i < size; i++ ) {
    sum= sum + a[i] * b[i];
  }
  // Vector version for remaining ops:
  // for(; i < size; i++ )
  // {
  //   vTemp= _mm_load_ss (&a[i]);
  //   vTemp2= _mm_load_ss (&b[i]);
  //   vTemp= _mm_mul_ps (vTemp, vTemp2);
  //   vSum= _mm_add_ps (vSum, vTemp);
  //   sum+= hsum_ps_sse3(vSum);
  // }
  return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void partA_routine2(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1 - (1.0/(b[i]+1.0));
  }
}

// in the following, size can have any positive value
void partA_vectorized2(float * restrict a, float * restrict b, int size) {
  __m128 vOnes= _mm_set1_ps(1.0);
  __m128 vTemp=  _mm_setzero_ps();
  int vecOps= ((int)(size / VECTOR_LEN))*VECTOR_LEN;
  int i= 0;
  for (; i < vecOps; i+= VECTOR_LEN)
  {
    vTemp= _mm_load_ps (&b[i]);
    vTemp= _mm_add_ps (vTemp, vOnes);
    vTemp= _mm_div_ps (vOnes, vTemp);
    vTemp= _mm_sub_ps (vOnes, vTemp);
    _mm_store_ps (&a[i], vTemp);
  }
  for (; i < size; i++ ) {
    a[i] = 1 - (1.0/(b[i]+1.0));
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void partA_routine3(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < 0.0 ) {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void partA_vectorized3(float * restrict a, float * restrict b, int size) {
  __m128 vZeros= _mm_setzero_ps();
  __m128 vA= _mm_setzero_ps();
  __m128 vB= _mm_setzero_ps();
  __m128 vIfs= _mm_setzero_ps();
  int vecOps= ((int)(size / VECTOR_LEN))*VECTOR_LEN;
  int i= 0;
  for (; i < vecOps; i+= 4 ) {
    vA= _mm_load_ps (&a[i]);
    vB= _mm_load_ps (&b[i]);
    vIfs= _mm_cmplt_ps (vA, vZeros);
    vB= _mm_and_ps (vIfs, vB);
    vA= _mm_andnot_ps (vIfs, vA);
    vA= _mm_or_ps (vA, vB);
    _mm_store_ps (&a[i], vA);
  }
  for (; i < size; i++ ) {
    if ( a[i] < 0.0 ) {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void partA_routine4(float * restrict a, float * restrict b,
		       float * restrict c) {
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i] - b[i+1]*c[i+1];
    a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
  }
}

void partA_vectorized4(float * restrict a, float * restrict b,
		       float * restrict  c) {
  __m128 vA1, vA2, vB1, vB2, vC1, vC2;
  __m128 temp1, temp2, temp3, temp4, temp5;
  __m128 vB1sts, vC1sts; //vector of every 1nd value from vX1 & vX2
  __m128 vB2nds, vC2nds; //vector of every 2nd value from vX1 & vX2
  __m128 vZeros= _mm_setzero_ps();
  for ( int i = 0; i < 2048; i = i+8  ) {
    vB1= _mm_load_ps (&b[i]);
    vB2= _mm_load_ps (&b[i+VECTOR_LEN]);
    vC1= _mm_load_ps (&c[i]);
    vC2= _mm_load_ps (&c[i+VECTOR_LEN]);

    temp1= _mm_moveldup_ps (vB1);
    temp1= _mm_shuffle_ps(temp1, vZeros, 2);
    temp1= _mm_shuffle_ps(temp1, vZeros, 1);
    temp2= _mm_moveldup_ps (vB2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    vB1sts= _mm_movelh_ps (temp1, temp2);

    temp1= _mm_moveldup_ps (vC1);
    temp1= _mm_shuffle_ps(temp1, vZeros, 2);
    temp1= _mm_shuffle_ps(temp1, vZeros, 1);
    temp2= _mm_moveldup_ps (vC2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    vC1sts= _mm_movelh_ps (temp1, temp2);

    temp1= _mm_movehdup_ps (vB1);
    temp1= _mm_shuffle_ps(temp1, vZeros, 2);
    temp1= _mm_shuffle_ps(temp1, vZeros, 1);
    temp2= _mm_movehdup_ps (vB2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    vB2nds= _mm_movelh_ps (temp1, temp2);

    temp1= _mm_movehdup_ps (vC1);
    temp1= _mm_shuffle_ps(temp1, vZeros, 2);
    temp1= _mm_shuffle_ps(temp1, vZeros, 1);
    temp2= _mm_movehdup_ps (vC2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    vC2nds= _mm_movelh_ps (temp1, temp2);

    temp1= _mm_mul_ps (vB1sts, vC1sts);
    temp2= _mm_mul_ps (vB2nds, vC2nds);
    temp4= _mm_sub_ps (temp1, temp2);
    temp1= _mm_mul_ps (vB1sts, vC2nds);
    temp2= _mm_mul_ps (vB2nds, vC1sts);
    temp5= _mm_add_ps (temp1, temp2);

    temp1= _mm_movelh_ps (temp4, temp5);
    temp1= _mm_shuffle_ps(temp1, vZeros, 2);
    temp1= _mm_shuffle_ps(temp1, vZeros, 1);
    temp2= _mm_movelh_ps (temp4, temp5);
    temp3= _mm_movehl_ps (vZeros, temp2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    temp2= _mm_movelh_ps (temp2, temp3);
    temp2= _mm_shuffle_ps(temp2, vZeros, 3);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    vA1= _mm_movelh_ps (temp1, temp2);

    temp1= _mm_movehl_ps (temp5, temp4);
    temp1= _mm_shuffle_ps(temp1, vZeros, 2);
    temp1= _mm_shuffle_ps(temp1, vZeros, 1);
    temp2= _mm_movehl_ps (temp5, temp4);
    temp3= _mm_movehl_ps (vZeros, temp2);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    temp2= _mm_movelh_ps (temp2, temp3);
    temp2= _mm_shuffle_ps(temp2, vZeros, 3);
    temp2= _mm_shuffle_ps(temp2, vZeros, 1);
    vA2= _mm_movelh_ps (temp1, temp2);
    
    _mm_store_ps (&a[i], vA1);
    _mm_store_ps (&a[i+VECTOR_LEN], vA2);
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void partA_routine5(unsigned char * restrict a,
		    unsigned char * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = b[i];
  }
}

void partA_vectorized5(unsigned char * restrict a,
		       unsigned char * restrict b, int size) {
  __m128i val;
  __m128i* zero= malloc(sizeof(__m128i));
  __m128i zerosMask= _mm_loadu_si64 (zero);
  int i= 0;
  //printf("\nzerosMask[%u-%u]: %lld %lld %lld %lld\n", i, i+3, zerosMask[0], zerosMask[1], zerosMask[2], zerosMask[3]);
  // for ( int i = 0; i < size; i+=16 ) {
  //   __m128i val1= _mm_set_epi8 (b[i], b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7], b[i+8], b[i+9], b[i+10], b[i+11], b[i+12], b[i+13], b[i+14], b[i+15]);
  //   _mm_maskmoveu_si128 (val1, zerosMask, (char*)&b[i]);
  //   printf("\nval1: %lld %lld %lld %lld",val1[0],val1[1],val1[2],val1[3]);
  //   a[i] = b[i];
  // }
}

/********************* routine 6 ***********************/

void partA_routine6(float * restrict a, float * restrict b,
		       float * restrict c) {
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void partA_vectorized6(float * restrict a, float * restrict b,
		       float * restrict c) {
  // replace the following code with vectorized code
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}



