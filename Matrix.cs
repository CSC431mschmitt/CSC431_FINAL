using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


/*
 * This code is distributed under the BSD license, and it is a rewrite of
 * code shared in CSC431 class at DePaul University by Massimo Di Pierro.
 */ 
namespace MatrixLib
{
    public class Matrix
    {
        private int _rows;
        private int _cols;
        private List<double[]> _data;
        
        public Matrix(int rows = 1, int cols = 1, double fill = 0.0)
        /* Constructor: Matrix
         * Purpose:     Constructs zero matrix 
         * Parameters:  rows - the integer number of rows
         *              cols - the integer number of columns
         *              fill - the value or callable to be used to fill the matrix
         */
        {
            _rows = rows;
            _cols = cols;
            _data = new List<double[]>();

            for (int i = 0; i < rows; i++)
            {
                double[] values = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    values[j] = fill;
                }
                _data.Add(values);
            }
        }

        public int rows
        {
            get { return _rows; }
            set { _rows = value; }
        }

        public int cols
        {
            get { return _cols; }
            set { _cols = value; }
        }

        public double[] data()
        {
            List<double> result = new List<double>();

            for (int i = 0; i < _data.Count(); i++)
            {
                for (int j = 0; j < _data[i].Count(); j++)
                {
                    result.Add(_data[i][j]);
                }
            }

            return result.ToArray();
        }

        public double this[int row, int column]
        {
            get { return _data[row][column]; }
            set { _data[row][column] = value; }
        }

        public Matrix Transpose()
        {
            Matrix T = new Matrix(_cols, _rows);
                            
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    T[j, i] = this[i, j];
                }
            }

            return T;
        }

        public Matrix Clone()
        {
            Matrix T = new Matrix(_rows, _cols);

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    T[i, j] = this[i, j];
                }
            }

            return T;
        }

        public List<double[]> as_list()
        /* 
         * Purpose:     Convert Matrix to List of double arrays, where each list item represents row in Matrix.
         * Parameters:  MxN Matrix
         * Returns:     List representation of Matrix.
         */
        {
            List<double[]> toList = new List<double[]>();
            double[] values;

            for (int r = 0; r < this._rows; r++)
            {
                values = new double[this._cols];

                for (int c = 0; c < _cols; c++)
                {
                    values[c] = this[r, c];
                }
                toList.Add(values);
            }
            
            return toList;
        }

        public List<double[]> as_()
        /* 
         * Purpose:     Convert Matrix to List of double arrays, where each list item represents row in Matrix.
         * Parameters:  MxN Matrix
         * Returns:     List representation of Matrix.
         */
        {
            List<double[]> toList = new List<double[]>();
            double[] values;

            for (int r = 0; r < this._rows; r++)
            {
                values = new double[this._cols];

                for (int c = 0; c < _cols; c++)
                {
                    values[c] = this[r, c];
                }
                toList.Add(values);
            }

            return toList;
        }

        public static Matrix from_list(List<double[]> vlist)
        /*
         * Purpose:     Builds a matrix from a list of arrays
         * Parameters:  List of arrays
         * Output:      Matrix with rows = number of lists, cols = number of items in first array
         */
        {
            int rows = vlist.Count();
            int cols = (vlist[0]).Length;

            Matrix M = new Matrix(rows, cols);

            for (int i = 0; i < vlist.Count; i++)
            {
                for (int j = 0; j < vlist[i].Length; j++)
                {
                    M[i,j] = vlist[i][j];
                }
            }
            
            return M;
        }

        public override string ToString()
        /* Override:    ToString
         * Purpose:     Prints out the full string of a Matrix in a readable row by row format.
         * Parameters:  None
         * Returns:     String representation of matrix
         */
        {
            String mFormat = "[";

            for (int i = 0; i < _rows; i++)
            {
                if (i != 0) mFormat += ", ";
                mFormat += "[";
                for (int j = 0; j < _cols; j++)
                {
                    if (j != 0) mFormat += ", ";
                    mFormat += (_data[i][j]).ToString();
                }
                mFormat += "]";
            }
            mFormat += "]";
            return mFormat;
        }

        public Matrix row(int row_num)
        /*
         * Parameters:  Row number of a Matrix.
         * Returns:     1xN Matrix of items in the indicated row_num of Matrix.
         */
        {
            Matrix M = new Matrix(1, this._cols);

            for (int c = 0; c < this._cols; c++)
            {
                M[0, c] = this[row_num, c];
            }
            
            return M;
        }
        
        public Matrix col(int col_num)
        /*
         * Parameters:  Column number of a Matrix.
         * Output:      Mx1 Matrix of the items in the indicated col_num of Matrix.
         */
        {
            Matrix M = new Matrix(this._rows, 1);

            for (int r = 0; r < this._rows; r++)
            {
                M[r, 0] = this[r, col_num];
            }

            return M;
        }

        public static Matrix identity(int rows, double diagonal = 1.0, double fill = 0.0)
        /* Constructor: Identity 
         * Purpose:     Constructs an n x n Matrix with the diagonal elements equal to the same value.  All non-diagonal elements are equal as well.
         *              Inherits from the Matrix class.
         * Parameters:  rows - the integer number of rows and columns.
         *              diagonal - the value of all diagonal elements
         *              fill - the value to be used to fill the non-diagonal elements of the identity matrix
         * Output:      rowsXrows square matrix with ones on the main diagonal and zeros elsewhere.
         */
        {
            Matrix M = new Matrix(rows, rows, fill);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    if (i == j)
                        M[i, j] = diagonal;
                    else
                        M[i, j] = fill;
                }
            }

            return M;
        }

        public static Matrix diagonal(double[] d)
        {
        /* Constructor: Diagonal 
         * Purpose:     Constructs an n x n Matrix with the i-th diagonal element equal to the i-th value of the passed array.
         *              Inherits from the Matrix class.M[i, i] = d[i]
         * Parameters:  d - array of elements to populate in diagonal elements of Matrix. 
         * Output:      Matrix where main diagonol entries M[i, i] = d[i], all other entries are 0.
         */
            
            int rows = d.Length;
            int cols = d.Length;
            Matrix M = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (i == j)
                        M[i, j] = d[i];
                    else
                        M[i, j] = 0.0;
                }
            }

            return M;
        }

        public static Matrix operator *(Matrix A, Matrix B)
        /* Override:    Multiplication (*)
         * Purpose:     Multiply two matrices A and B together to return matrix M. M[i,j] = A_i1*B_1j + A_i2*B_2j + ... + A_in*B_nj.
         *              Exception will thrown if number of columns of matrix A does not equal the number of rows in matrix B.
         * Parameters:  A - First matrix operand.
         *              B - Second matrix operand.
         * Returns:     Resulting matrix of the matrix multiplication.
         */
        {
            Matrix M = null;
            
            try
            {
                if ( (A._cols == 1 && B._cols == 1) && (A._rows == B._rows) )
                {
                    M = new Matrix(1, 1);
                    for (int r = 0; r < A._rows; r++)
                    {
                        M[0,0] += A[r, 0] * B[r, 0];
                    }
                }
                else if (A._cols != B._rows)
                {
                    throw new InvalidOperationException("Incompatible Dimensions: Matrix A.cols != Matrix B.rows");
                }
                else
                {
                    M = new Matrix(A._rows, B._cols);

                    for (int r = 0; r < A._rows; r++)
                    {
                        for (int c = 0; c < B._cols; c++)
                        {
                            for (int k = 0; k < A._cols; k++)
                                M[r, c] = M[r, c] + (A[r, k] * B[k, c]);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return M;
        }

        public static Matrix operator *(Matrix A, double x)
        /* Override:    Multiplication (*)
         * Purpose:     Multiply Matrix A and scalar B together to return matrix M. M[i,j] = A_ij * x
         * Parameters:  A - Matrix
         *              B - Scalar
         * Returns:     Resulting matrix of the matrix by scalar multiplication.
         */
        {
            
            Matrix M = new Matrix(A._rows, A._cols);

            for (int r = 0; r < A._rows; r++)
            {
                for (int c = 0; c < A._cols; c++)
                    M[r, c] = A[r, c] * x;
            }

            return M;
        }

        public static Matrix operator /(double x, Matrix A)
        /* Override:    Division (/)
         * Purpose:     Computes x/A using Gauss-Jordan elimination.
         *              Exception thrown if Matrix A is singular.
         * Parameters:  A - Matrix.
         *              x - Scalar.
         * Returns:     Resulting matrix of the scalar by matrix division.
         */
        {
            Matrix B = null;

            int cols = A._cols;

            try
            {
                if (A._rows != cols)
                {
                    throw new ArithmeticException("Matrix A not squared.");
                }
                else
                {
                    int upper_index = cols - 1;
                    
                    A = A.Clone();
                    B = Matrix.identity(cols, x);

                    for (int c = 0; c < cols; c++)
                    {
                        for (int r = c+1; r < cols; r++)
                        {
                            if (Math.Abs(A[r, c]) > Math.Abs(A[c, c]))
                            {
                                A.swap_rows(r, c);
                                B.swap_rows(r, c);
                            }
                        }
                        double p = 0.0 + A[c, c]; //trick to implicitly cast as double

                        for (int k = 0; k < cols; k++)
                        {
                            A[c, k] = A[c, k] / p;
                            B[c, k] = B[c, k] / p;
                        }

                        for(int r=0; r < cols; r++) if(r!=c) 
                        {
                            p = 0.0 + A[r, c]; //trick to implicitly cast as double
                            for (int k = 0; k < cols; k++)
                            {
                                A[r, k] -= A[c, k] * p;
                                B[r, k] -= B[c, k] * p;
                            }
                        }
                    }
                }
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return B;
        }

        public static Matrix operator /(Matrix A, double x)
        /* Override:    Division (/)
         * Purpose:     Computes A/x by multiplying A by the inverse of x.
         *              Exception thrown if Matrix A is singular.
         * Parameters:  A - Matrix.
         *              x - Scalar.
         * Returns:     Resulting matrix of the matrix by scalar division.
         */
        {
            return A * (1.0/x);
        }

        public static Matrix operator /(Matrix A, Matrix B)
        /* Override:    Division (/)
         * Purpose:     Computes A/B using Gauss-Jordan elimination using scalar of 1.
         *              Exception thrown if Matrix A is singular.
         * Parameters:  A - Matrix.
         *              B - Matrix.
         * Returns:     Resulting matrix of the matrix division.
         */
        {
            Matrix M = null;

            try
            {
                if (B._rows != B._cols)
                {
                    throw new ArithmeticException("Matrix B is not square.");
                }
                else
                {
                    M = A*(1.0 / B);
                }
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return M;
        }

        public void swap_rows(int i, int j)
        {
            try
            {
                //var ex = new InvalidOperationException("Incompatible Dimensions");

                if ((this._data).Count != 2) 
                {
                    throw new InvalidOperationException("Incompatible Dimensions");
                }

                else if ((i >= this._rows) || (j >= this._rows))
                {
                    throw new InvalidOperationException("Index out of bounds");
                }

                else
                {
                    for (int c = 0; c < this._cols; c++)
                    {
                        double x = this[i, c];
                        this[i, c] = this[j, c];
                        this[j, c] = x;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        public static Matrix operator +(Matrix A, Matrix B)
        /* Override:    Addition (+)
         * Purpose:     Add two matrices A and B together to return matrix M. M[i,j] = A[i,j] + B[i,j]
         *              Exception thrown if matrix A is not equal in size to matrix B.
         * Parameters:  A - First matrix operand.
         *              B - Second matrix operand.
         * Returns:     Resulting matrix of the matrix addition.
         */
        {
            int rows = A._rows;
            int cols = A._cols;
            Matrix M = null;

            try
            {
                if ((B._rows != rows) || (B._cols != cols))
                {
                    throw new InvalidOperationException("Incompatible Dimensions: Matrix A size != Matrix B size");
                }
                else
                {
                    M = new Matrix(rows, cols);

                    for (int i = 0; i < A._rows; i++)
                    {
                        for (int j = 0; j < A._cols; j++)
                        {
                            M[i, j] = A[i, j] + B[i, j];
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
                        
            return M;
        }

        public static Matrix operator -(Matrix A)
        /* Override:    Negative (-)
         * Purpose:     Reverse the sign of each element in matrix A. A[i,j] = -A[i,j]
         * Parameters:  A - Matrix.
         * Returns:     Resulting matrix of the opposite sign.
         */
        {
            for (int i = 0; i < A._rows; i++)
                for (int j = 0; j < A._cols; j++)
                    A[i, j] *= -1;
            return A;
        }

        public static Matrix operator -(Matrix A, double x)
        /* Override:    Subtraction (-)
         * Purpose:     Subtract a scalar from each element in matrix A. A[i,j] = A[i,j] - x
         * Parameters:  A - Matrix.
         *              x - scalar
         * Returns:     Resulting matrix after the subtraction operation.
         */
        {
            for (int i = 0; i < A._rows; i++)
                for (int j = 0; j < A._cols; j++)
                    A[i, j] -= x;
            return A;
        }

        public static Matrix operator +(Matrix A, double x)
        /* Override:    Addition (+)
         * Purpose:     Add a scalar to each element in matrix A. A[i,j] = A[i,j] + x
         * Parameters:  A - Matrix.
         *              x - scalar
         * Returns:     Resulting matrix after the addition operation.
         */
        {
            for (int i = 0; i < A._rows; i++)
                for (int j = 0; j < A._cols; j++)
                    A[i, j] += x;
            return A;
        }

        public static Matrix operator -(Matrix A, Matrix B)
        /* Override:    Subtraction (-)
         * Purpose:     Subtract two matrices A and B to return matrix M. M[i,j] = A[i,j] - B[i,j]
         *              Exception thrown if matrix A is not equal in size to matrix B.
         * Parameters:  A - First matrix operand.
         *              B - Second matrix operand.
         * Returns:     Resulting matrix of the matrix subtraction.
         */
        {
            int rows = A._rows;
            int cols = A._cols;
            Matrix M = null;

            try
            {
                if ((B._rows != rows) || (B._cols != cols))
                {
                    throw new InvalidOperationException("Incompatible Dimensions: Matrix A size != Matrix B size");
                }
                else
                {
                    M = new Matrix(rows, cols);

                    for (int i = 0; i < A._rows; i++)
                    {
                        for (int j = 0; j < A._cols; j++)
                        {
                            M[i, j] = A[i, j] - B[i, j];
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return M;
        }

        public double norm()
        {
            double max_value = 0, col_sum;
            
            for (int i = 0; i < this.cols; i++)
            {
                col_sum = 0;
                for (int j = 0; j < this.rows; j++)
                    col_sum += this[j, i];

                if (col_sum > max_value)
                    max_value = col_sum;
            }

            return max_value;
        }

    }
}
