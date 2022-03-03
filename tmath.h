#include <cmath>
#include <vector>
#include <iostream>
#include <complex>
#include <chrono>
#include <iomanip>
using namespace std;



// Vertex class, using a type (e.g. float) and a vertex component count (e.g. 8 for octonions)
template<class T, size_t N>
class vertex
{
public:
	vector<T> vd; // vertex data

	vertex(void)
	{
		vd.resize(N, 0); // allocate the memory
	}

	bool operator==(const vertex& rhs) const
	{
		bool all_equal = true;

		for (size_t i = 0; i < N; i++)
		{
			T f = fabs(vd[i] - rhs.vd[i]);

			if (f > 0.0001)
			{
				all_equal = false;
				break;
			}
		}

		return all_equal;
	}

	bool operator!=(const vertex& rhs) const
	{
		return !(*this == rhs);
	}

	vertex operator+(const vertex& rhs) const
	{
		vertex out;

		for (size_t i = 0; i < N; i++)
			out.vd[i] = vd[i] + rhs.vd[i];

		return out;
	}

	vertex operator*(const T rhs) const
	{
		vertex out;

		for (size_t i = 0; i < N; i++)
			out.vd[i] = vd[i] * rhs;

		return out;
	}

	T magnitude(void) const
	{
		return sqrt(all_dot(*this));
	}

	T all_dot(const vertex& rhs) const
	{
		T all_self_dot = 0;

		for (size_t i = 0; i < N; i++)
			all_self_dot += (vd[i] * rhs.vd[i]);

		return all_self_dot;
	}

	T imag_dot(const vertex& rhs) const
	{
		T imag_self_dot = 0;

		for (size_t i = 1; i < N; i++)
			imag_self_dot += (vd[i] * rhs.vd[i]);

		return imag_self_dot;
	}
};

template<class T, size_t N>
vertex<T, N> pow(vertex<T, N>& in, vertex<T, N>& exponent)
{
	vertex<T, N> ln_a = log(log(in));
	vertex<T, N> ln_b = log(exponent);

	vertex<T, N> sum = ln_a + ln_b;
	vertex<T, N> out = exp(sum);

	return exp(out);
}

// Pow function for variable T and N
template<class T, size_t N>
vertex<T, N> pow(const vertex<T, N>& in, T beta)
{
//	return exp(log(in) * beta); // slower, but gets the point across

	T all_self_dot = 0;
	T imag_self_dot = 0;
	vertex<T, N> out;

	for (size_t i = 1; i < N; i++)
		imag_self_dot += (in.vd[i] * in.vd[i]);

	all_self_dot = imag_self_dot + (in.vd[0] * in.vd[0]);

	if (all_self_dot == 0)
	{
		for (size_t i = 0; i < N; i++)
			out.vd[i] = 0;

		return out;
	}

	const T l_d = sqrt(all_self_dot);
	const T l_e = sqrt(imag_self_dot);
	const T self_dot_beta = pow(l_d, beta);

	out.vd[0] = self_dot_beta * cos(beta * acos(in.vd[0] / l_d));

	if (l_e != 0)
	{
		const T x = 1 / l_e;
		const T y = self_dot_beta * sin(beta * acos(in.vd[0] / l_d));
		const T z = x * y;

		for (size_t i = 1; i < N; i++)	
			out.vd[i] = in.vd[i] * z;
	}

	return out;
}

template<class T, size_t N>
vertex<T, N> square(const vertex<T, N>& in)
{
	vertex<T, N> out;

	out.vd[0] = in.vd[0] * in.vd[0];

	for (size_t i = 1; i < N; i++)
	{
		out.vd[0] -= in.vd[i] * in.vd[i];
		out.vd[i] = 2 * in.vd[0] * in.vd[i];
	}

	return out;
}

// Traditional multiplication for n = 1 (e.g. real numbers) for variable T
template<class T, size_t N = 1>
vertex<T, 1> traditional_mul(const vertex<T, 1>& in_a, const vertex<T, 1>& in_b)
{
	vertex<T, 1> out;

	out.vd[0] = in_a.vd[0] * in_b.vd[0];

	return out;
}

// Traditional multiplication for n = 2 (e.g. complex numbers) for variable T
template<class T, size_t N = 2>
vertex<T, 2> traditional_mul(const vertex<T, 2>& in_a, const vertex<T, 2>& in_b)
{
	vertex<T, 2> out;

	out.vd[0] = in_a.vd[0] * in_b.vd[0] - in_a.vd[1] * in_b.vd[1];
	out.vd[1] = in_a.vd[0] * in_b.vd[1] + in_a.vd[1] * in_b.vd[0];

	return out;
}

// Traditional multiplication for n = 4 (e.g. quaternions) for variable T
template<class T, size_t N = 4>
vertex<T, 4> traditional_mul(const vertex<T, 4>& in_a, const vertex<T, 4>& in_b)
{
	vertex<T, 4> out;

	out.vd[0] = in_a.vd[0] * in_b.vd[0] - in_a.vd[1] * in_b.vd[1] - in_a.vd[2] * in_b.vd[2] - in_a.vd[3] * in_b.vd[3];
	out.vd[1] = in_a.vd[0] * in_b.vd[1] + in_a.vd[1] * in_b.vd[0] + in_a.vd[2] * in_b.vd[3] - in_a.vd[3] * in_b.vd[2];
	out.vd[2] = in_a.vd[0] * in_b.vd[2] - in_a.vd[1] * in_b.vd[3] + in_a.vd[2] * in_b.vd[0] + in_a.vd[3] * in_b.vd[1];
	out.vd[3] = in_a.vd[0] * in_b.vd[3] + in_a.vd[1] * in_b.vd[2] - in_a.vd[2] * in_b.vd[1] + in_a.vd[3] * in_b.vd[0];

	return out;
}

// Traditional multiplication for n = 8 (e.g. octonions) for variable T
template<class T, size_t N = 8>
vertex<T, 8> traditional_mul(const vertex<T, 8>& in_a, const vertex<T, 8>& in_b)
{
	vertex<T, 8> out;

	out.vd[0] = in_a.vd[0] * in_b.vd[0] - in_a.vd[1] * in_b.vd[1] - in_a.vd[2] * in_b.vd[2] - in_a.vd[3] * in_b.vd[3] - in_a.vd[4] * in_b.vd[4] - in_a.vd[5] * in_b.vd[5] - in_a.vd[6] * in_b.vd[6] - in_a.vd[7] * in_b.vd[7];
	out.vd[1] = in_a.vd[0] * in_b.vd[1] + in_a.vd[1] * in_b.vd[0] + in_a.vd[2] * in_b.vd[3] - in_a.vd[3] * in_b.vd[2] + in_a.vd[4] * in_b.vd[5] - in_a.vd[5] * in_b.vd[4] - in_a.vd[6] * in_b.vd[7] + in_a.vd[7] * in_b.vd[6];
	out.vd[2] = in_a.vd[0] * in_b.vd[2] - in_a.vd[1] * in_b.vd[3] + in_a.vd[2] * in_b.vd[0] + in_a.vd[3] * in_b.vd[1] + in_a.vd[4] * in_b.vd[6] + in_a.vd[5] * in_b.vd[7] - in_a.vd[6] * in_b.vd[4] - in_a.vd[7] * in_b.vd[5];
	out.vd[3] = in_a.vd[0] * in_b.vd[3] + in_a.vd[1] * in_b.vd[2] - in_a.vd[2] * in_b.vd[1] + in_a.vd[3] * in_b.vd[0] + in_a.vd[4] * in_b.vd[7] - in_a.vd[5] * in_b.vd[6] + in_a.vd[6] * in_b.vd[5] - in_a.vd[7] * in_b.vd[4];
	out.vd[4] = in_a.vd[0] * in_b.vd[4] - in_a.vd[1] * in_b.vd[5] - in_a.vd[2] * in_b.vd[6] - in_a.vd[3] * in_b.vd[7] + in_a.vd[4] * in_b.vd[0] + in_a.vd[5] * in_b.vd[1] + in_a.vd[6] * in_b.vd[2] + in_a.vd[7] * in_b.vd[3];
	out.vd[5] = in_a.vd[0] * in_b.vd[5] + in_a.vd[1] * in_b.vd[4] - in_a.vd[2] * in_b.vd[7] + in_a.vd[3] * in_b.vd[6] - in_a.vd[4] * in_b.vd[1] + in_a.vd[5] * in_b.vd[0] - in_a.vd[6] * in_b.vd[3] + in_a.vd[7] * in_b.vd[2];
	out.vd[6] = in_a.vd[0] * in_b.vd[6] + in_a.vd[1] * in_b.vd[7] + in_a.vd[2] * in_b.vd[4] - in_a.vd[3] * in_b.vd[5] - in_a.vd[4] * in_b.vd[2] + in_a.vd[5] * in_b.vd[3] + in_a.vd[6] * in_b.vd[0] - in_a.vd[7] * in_b.vd[1];
	out.vd[7] = in_a.vd[0] * in_b.vd[7] - in_a.vd[1] * in_b.vd[6] + in_a.vd[2] * in_b.vd[5] + in_a.vd[3] * in_b.vd[4] - in_a.vd[4] * in_b.vd[3] - in_a.vd[5] * in_b.vd[2] + in_a.vd[6] * in_b.vd[1] + in_a.vd[7] * in_b.vd[0];

	return out;
}

// Traditional multiplication for n = 16 (e.g. sedenions) for variable T
template<class T, size_t N = 16>
vertex<T, 16> traditional_mul(const vertex<T, 16>& in_a, const vertex<T, 16>& in_b)
{
	vertex<T, 16> out;

	out.vd[0] = in_a.vd[0] * in_b.vd[0] - in_a.vd[1] * in_b.vd[1] - in_a.vd[2] * in_b.vd[2] - in_a.vd[3] * in_b.vd[3] - in_a.vd[4] * in_b.vd[4] - in_a.vd[5] * in_b.vd[5] - in_a.vd[6] * in_b.vd[6] - in_a.vd[7] * in_b.vd[7] - in_a.vd[8] * in_b.vd[8] - in_a.vd[9] * in_b.vd[9] - in_a.vd[10] * in_b.vd[10] - in_a.vd[11] * in_b.vd[11] - in_a.vd[12] * in_b.vd[12] - in_a.vd[13] * in_b.vd[13] - in_a.vd[14] * in_b.vd[14] - in_a.vd[15] * in_b.vd[15];
	out.vd[1] = in_a.vd[0] * in_b.vd[1] + in_a.vd[1] * in_b.vd[0] + in_a.vd[2] * in_b.vd[3] - in_a.vd[3] * in_b.vd[2] + in_a.vd[4] * in_b.vd[5] - in_a.vd[5] * in_b.vd[4] - in_a.vd[6] * in_b.vd[7] + in_a.vd[7] * in_b.vd[6] + in_a.vd[8] * in_b.vd[9] - in_a.vd[9] * in_b.vd[8] - in_a.vd[10] * in_b.vd[11] + in_a.vd[11] * in_b.vd[10] - in_a.vd[12] * in_b.vd[13] + in_a.vd[13] * in_b.vd[12] + in_a.vd[14] * in_b.vd[15] - in_a.vd[15] * in_b.vd[14];
	out.vd[2] = in_a.vd[0] * in_b.vd[2] - in_a.vd[1] * in_b.vd[3] + in_a.vd[2] * in_b.vd[0] + in_a.vd[3] * in_b.vd[1] + in_a.vd[4] * in_b.vd[6] + in_a.vd[5] * in_b.vd[7] - in_a.vd[6] * in_b.vd[4] - in_a.vd[7] * in_b.vd[5] + in_a.vd[8] * in_b.vd[10] + in_a.vd[9] * in_b.vd[11] - in_a.vd[10] * in_b.vd[8] - in_a.vd[11] * in_b.vd[9] - in_a.vd[12] * in_b.vd[14] - in_a.vd[13] * in_b.vd[15] + in_a.vd[14] * in_b.vd[12] + in_a.vd[15] * in_b.vd[13];
	out.vd[3] = in_a.vd[0] * in_b.vd[3] + in_a.vd[1] * in_b.vd[2] - in_a.vd[2] * in_b.vd[1] + in_a.vd[3] * in_b.vd[0] + in_a.vd[4] * in_b.vd[7] - in_a.vd[5] * in_b.vd[6] + in_a.vd[6] * in_b.vd[5] - in_a.vd[7] * in_b.vd[4] + in_a.vd[8] * in_b.vd[11] - in_a.vd[9] * in_b.vd[10] + in_a.vd[10] * in_b.vd[9] - in_a.vd[11] * in_b.vd[8] - in_a.vd[12] * in_b.vd[15] + in_a.vd[13] * in_b.vd[14] - in_a.vd[14] * in_b.vd[13] + in_a.vd[15] * in_b.vd[12];
	out.vd[4] = in_a.vd[0] * in_b.vd[4] - in_a.vd[1] * in_b.vd[5] - in_a.vd[2] * in_b.vd[6] - in_a.vd[3] * in_b.vd[7] + in_a.vd[4] * in_b.vd[0] + in_a.vd[5] * in_b.vd[1] + in_a.vd[6] * in_b.vd[2] + in_a.vd[7] * in_b.vd[3] + in_a.vd[8] * in_b.vd[12] + in_a.vd[9] * in_b.vd[13] + in_a.vd[10] * in_b.vd[14] + in_a.vd[11] * in_b.vd[15] - in_a.vd[12] * in_b.vd[8] - in_a.vd[13] * in_b.vd[9] - in_a.vd[14] * in_b.vd[10] - in_a.vd[15] * in_b.vd[11];
	out.vd[5] = in_a.vd[0] * in_b.vd[5] + in_a.vd[1] * in_b.vd[4] - in_a.vd[2] * in_b.vd[7] + in_a.vd[3] * in_b.vd[6] - in_a.vd[4] * in_b.vd[1] + in_a.vd[5] * in_b.vd[0] - in_a.vd[6] * in_b.vd[3] + in_a.vd[7] * in_b.vd[2] + in_a.vd[8] * in_b.vd[13] - in_a.vd[9] * in_b.vd[12] + in_a.vd[10] * in_b.vd[15] - in_a.vd[11] * in_b.vd[14] + in_a.vd[12] * in_b.vd[9] - in_a.vd[13] * in_b.vd[8] + in_a.vd[14] * in_b.vd[11] - in_a.vd[15] * in_b.vd[10];
	out.vd[6] = in_a.vd[0] * in_b.vd[6] + in_a.vd[1] * in_b.vd[7] + in_a.vd[2] * in_b.vd[4] - in_a.vd[3] * in_b.vd[5] - in_a.vd[4] * in_b.vd[2] + in_a.vd[5] * in_b.vd[3] + in_a.vd[6] * in_b.vd[0] - in_a.vd[7] * in_b.vd[1] + in_a.vd[8] * in_b.vd[14] - in_a.vd[9] * in_b.vd[15] - in_a.vd[10] * in_b.vd[12] + in_a.vd[11] * in_b.vd[13] + in_a.vd[12] * in_b.vd[10] - in_a.vd[13] * in_b.vd[11] - in_a.vd[14] * in_b.vd[8] + in_a.vd[15] * in_b.vd[9];
	out.vd[7] = in_a.vd[0] * in_b.vd[7] - in_a.vd[1] * in_b.vd[6] + in_a.vd[2] * in_b.vd[5] + in_a.vd[3] * in_b.vd[4] - in_a.vd[4] * in_b.vd[3] - in_a.vd[5] * in_b.vd[2] + in_a.vd[6] * in_b.vd[1] + in_a.vd[7] * in_b.vd[0] + in_a.vd[8] * in_b.vd[15] + in_a.vd[9] * in_b.vd[14] - in_a.vd[10] * in_b.vd[13] - in_a.vd[11] * in_b.vd[12] + in_a.vd[12] * in_b.vd[11] + in_a.vd[13] * in_b.vd[10] - in_a.vd[14] * in_b.vd[9] - in_a.vd[15] * in_b.vd[8];
	out.vd[8] = in_a.vd[0] * in_b.vd[8] - in_a.vd[1] * in_b.vd[9] - in_a.vd[2] * in_b.vd[10] - in_a.vd[3] * in_b.vd[11] - in_a.vd[4] * in_b.vd[12] - in_a.vd[5] * in_b.vd[13] - in_a.vd[6] * in_b.vd[14] - in_a.vd[7] * in_b.vd[15] + in_a.vd[8] * in_b.vd[0] + in_a.vd[9] * in_b.vd[1] + in_a.vd[10] * in_b.vd[2] + in_a.vd[11] * in_b.vd[3] + in_a.vd[12] * in_b.vd[4] + in_a.vd[13] * in_b.vd[5] + in_a.vd[14] * in_b.vd[6] + in_a.vd[15] * in_b.vd[7];
	out.vd[9] = in_a.vd[0] * in_b.vd[9] + in_a.vd[1] * in_b.vd[8] - in_a.vd[2] * in_b.vd[11] + in_a.vd[3] * in_b.vd[10] - in_a.vd[4] * in_b.vd[13] + in_a.vd[5] * in_b.vd[12] + in_a.vd[6] * in_b.vd[15] - in_a.vd[7] * in_b.vd[14] - in_a.vd[8] * in_b.vd[1] + in_a.vd[9] * in_b.vd[0] - in_a.vd[10] * in_b.vd[3] + in_a.vd[11] * in_b.vd[2] - in_a.vd[12] * in_b.vd[5] + in_a.vd[13] * in_b.vd[4] + in_a.vd[14] * in_b.vd[7] - in_a.vd[15] * in_b.vd[6];
	out.vd[10] = in_a.vd[0] * in_b.vd[10] + in_a.vd[1] * in_b.vd[11] + in_a.vd[2] * in_b.vd[8] - in_a.vd[3] * in_b.vd[9] - in_a.vd[4] * in_b.vd[14] - in_a.vd[5] * in_b.vd[15] + in_a.vd[6] * in_b.vd[12] + in_a.vd[7] * in_b.vd[13] - in_a.vd[8] * in_b.vd[2] + in_a.vd[9] * in_b.vd[3] + in_a.vd[10] * in_b.vd[0] - in_a.vd[11] * in_b.vd[1] - in_a.vd[12] * in_b.vd[6] - in_a.vd[13] * in_b.vd[7] + in_a.vd[14] * in_b.vd[4] + in_a.vd[15] * in_b.vd[5];
	out.vd[11] = in_a.vd[0] * in_b.vd[11] - in_a.vd[1] * in_b.vd[10] + in_a.vd[2] * in_b.vd[9] + in_a.vd[3] * in_b.vd[8] - in_a.vd[4] * in_b.vd[15] + in_a.vd[5] * in_b.vd[14] - in_a.vd[6] * in_b.vd[13] + in_a.vd[7] * in_b.vd[12] - in_a.vd[8] * in_b.vd[3] - in_a.vd[9] * in_b.vd[2] + in_a.vd[10] * in_b.vd[1] + in_a.vd[11] * in_b.vd[0] - in_a.vd[12] * in_b.vd[7] + in_a.vd[13] * in_b.vd[6] - in_a.vd[14] * in_b.vd[5] + in_a.vd[15] * in_b.vd[4];
	out.vd[12] = in_a.vd[0] * in_b.vd[12] + in_a.vd[1] * in_b.vd[13] + in_a.vd[2] * in_b.vd[14] + in_a.vd[3] * in_b.vd[15] + in_a.vd[4] * in_b.vd[8] - in_a.vd[5] * in_b.vd[9] - in_a.vd[6] * in_b.vd[10] - in_a.vd[7] * in_b.vd[11] - in_a.vd[8] * in_b.vd[4] + in_a.vd[9] * in_b.vd[5] + in_a.vd[10] * in_b.vd[6] + in_a.vd[11] * in_b.vd[7] + in_a.vd[12] * in_b.vd[0] - in_a.vd[13] * in_b.vd[1] - in_a.vd[14] * in_b.vd[2] - in_a.vd[15] * in_b.vd[3];
	out.vd[13] = in_a.vd[0] * in_b.vd[13] - in_a.vd[1] * in_b.vd[12] + in_a.vd[2] * in_b.vd[15] - in_a.vd[3] * in_b.vd[14] + in_a.vd[4] * in_b.vd[9] + in_a.vd[5] * in_b.vd[8] + in_a.vd[6] * in_b.vd[11] - in_a.vd[7] * in_b.vd[10] - in_a.vd[8] * in_b.vd[5] - in_a.vd[9] * in_b.vd[4] + in_a.vd[10] * in_b.vd[7] - in_a.vd[11] * in_b.vd[6] + in_a.vd[12] * in_b.vd[1] + in_a.vd[13] * in_b.vd[0] + in_a.vd[14] * in_b.vd[3] - in_a.vd[15] * in_b.vd[2];
	out.vd[14] = in_a.vd[0] * in_b.vd[14] - in_a.vd[1] * in_b.vd[15] - in_a.vd[2] * in_b.vd[12] + in_a.vd[3] * in_b.vd[13] + in_a.vd[4] * in_b.vd[10] - in_a.vd[5] * in_b.vd[11] + in_a.vd[6] * in_b.vd[8] + in_a.vd[7] * in_b.vd[9] - in_a.vd[8] * in_b.vd[6] - in_a.vd[9] * in_b.vd[7] - in_a.vd[10] * in_b.vd[4] + in_a.vd[11] * in_b.vd[5] + in_a.vd[12] * in_b.vd[2] - in_a.vd[13] * in_b.vd[3] + in_a.vd[14] * in_b.vd[0] + in_a.vd[15] * in_b.vd[1];
	out.vd[15] = in_a.vd[0] * in_b.vd[15] + in_a.vd[1] * in_b.vd[14] - in_a.vd[2] * in_b.vd[13] - in_a.vd[3] * in_b.vd[12] + in_a.vd[4] * in_b.vd[11] + in_a.vd[5] * in_b.vd[10] - in_a.vd[6] * in_b.vd[9] + in_a.vd[7] * in_b.vd[8] - in_a.vd[8] * in_b.vd[7] + in_a.vd[9] * in_b.vd[6] - in_a.vd[10] * in_b.vd[5] - in_a.vd[11] * in_b.vd[4] + in_a.vd[12] * in_b.vd[3] + in_a.vd[13] * in_b.vd[2] - in_a.vd[14] * in_b.vd[1] + in_a.vd[15] * in_b.vd[0];

	return out;
}

// Traditional multiplication for n = 32 (e.g. pathions) for variable T
template<class T, size_t N = 32>
vertex<T, 32> traditional_mul(const vertex<T, 32>& in_a, const vertex<T, 32>& in_b)
{
	vertex<T, 32> out;

	out.vd[0] = in_a.vd[0] * in_b.vd[0] - in_a.vd[1] * in_b.vd[1] - in_a.vd[2] * in_b.vd[2] - in_a.vd[3] * in_b.vd[3] - in_a.vd[4] * in_b.vd[4] - in_a.vd[5] * in_b.vd[5] - in_a.vd[6] * in_b.vd[6] - in_a.vd[7] * in_b.vd[7] - in_a.vd[8] * in_b.vd[8] - in_a.vd[9] * in_b.vd[9] - in_a.vd[10] * in_b.vd[10] - in_a.vd[11] * in_b.vd[11] - in_a.vd[12] * in_b.vd[12] - in_a.vd[13] * in_b.vd[13] - in_a.vd[14] * in_b.vd[14] - in_a.vd[15] * in_b.vd[15] - in_a.vd[16] * in_b.vd[16] - in_a.vd[17] * in_b.vd[17] - in_a.vd[18] * in_b.vd[18] - in_a.vd[19] * in_b.vd[19] - in_a.vd[20] * in_b.vd[20] - in_a.vd[21] * in_b.vd[21] - in_a.vd[22] * in_b.vd[22] - in_a.vd[23] * in_b.vd[23] - in_a.vd[24] * in_b.vd[24] - in_a.vd[25] * in_b.vd[25] - in_a.vd[26] * in_b.vd[26] - in_a.vd[27] * in_b.vd[27] - in_a.vd[28] * in_b.vd[28] - in_a.vd[29] * in_b.vd[29] - in_a.vd[30] * in_b.vd[30] - in_a.vd[31] * in_b.vd[31];
	out.vd[1] = in_a.vd[0] * in_b.vd[1] + in_a.vd[1] * in_b.vd[0] + in_a.vd[2] * in_b.vd[3] - in_a.vd[3] * in_b.vd[2] + in_a.vd[4] * in_b.vd[5] - in_a.vd[5] * in_b.vd[4] - in_a.vd[6] * in_b.vd[7] + in_a.vd[7] * in_b.vd[6] + in_a.vd[8] * in_b.vd[9] - in_a.vd[9] * in_b.vd[8] - in_a.vd[10] * in_b.vd[11] + in_a.vd[11] * in_b.vd[10] - in_a.vd[12] * in_b.vd[13] + in_a.vd[13] * in_b.vd[12] + in_a.vd[14] * in_b.vd[15] - in_a.vd[15] * in_b.vd[14] + in_a.vd[16] * in_b.vd[17] - in_a.vd[17] * in_b.vd[16] - in_a.vd[18] * in_b.vd[19] + in_a.vd[19] * in_b.vd[18] - in_a.vd[20] * in_b.vd[21] + in_a.vd[21] * in_b.vd[20] + in_a.vd[22] * in_b.vd[23] - in_a.vd[23] * in_b.vd[22] - in_a.vd[24] * in_b.vd[25] + in_a.vd[25] * in_b.vd[24] + in_a.vd[26] * in_b.vd[27] - in_a.vd[27] * in_b.vd[26] + in_a.vd[28] * in_b.vd[29] - in_a.vd[29] * in_b.vd[28] - in_a.vd[30] * in_b.vd[31] + in_a.vd[31] * in_b.vd[30];
	out.vd[2] = in_a.vd[0] * in_b.vd[2] - in_a.vd[1] * in_b.vd[3] + in_a.vd[2] * in_b.vd[0] + in_a.vd[3] * in_b.vd[1] + in_a.vd[4] * in_b.vd[6] + in_a.vd[5] * in_b.vd[7] - in_a.vd[6] * in_b.vd[4] - in_a.vd[7] * in_b.vd[5] + in_a.vd[8] * in_b.vd[10] + in_a.vd[9] * in_b.vd[11] - in_a.vd[10] * in_b.vd[8] - in_a.vd[11] * in_b.vd[9] - in_a.vd[12] * in_b.vd[14] - in_a.vd[13] * in_b.vd[15] + in_a.vd[14] * in_b.vd[12] + in_a.vd[15] * in_b.vd[13] + in_a.vd[16] * in_b.vd[18] + in_a.vd[17] * in_b.vd[19] - in_a.vd[18] * in_b.vd[16] - in_a.vd[19] * in_b.vd[17] - in_a.vd[20] * in_b.vd[22] - in_a.vd[21] * in_b.vd[23] + in_a.vd[22] * in_b.vd[20] + in_a.vd[23] * in_b.vd[21] - in_a.vd[24] * in_b.vd[26] - in_a.vd[25] * in_b.vd[27] + in_a.vd[26] * in_b.vd[24] + in_a.vd[27] * in_b.vd[25] + in_a.vd[28] * in_b.vd[30] + in_a.vd[29] * in_b.vd[31] - in_a.vd[30] * in_b.vd[28] - in_a.vd[31] * in_b.vd[29];
	out.vd[3] = in_a.vd[0] * in_b.vd[3] + in_a.vd[1] * in_b.vd[2] - in_a.vd[2] * in_b.vd[1] + in_a.vd[3] * in_b.vd[0] + in_a.vd[4] * in_b.vd[7] - in_a.vd[5] * in_b.vd[6] + in_a.vd[6] * in_b.vd[5] - in_a.vd[7] * in_b.vd[4] + in_a.vd[8] * in_b.vd[11] - in_a.vd[9] * in_b.vd[10] + in_a.vd[10] * in_b.vd[9] - in_a.vd[11] * in_b.vd[8] - in_a.vd[12] * in_b.vd[15] + in_a.vd[13] * in_b.vd[14] - in_a.vd[14] * in_b.vd[13] + in_a.vd[15] * in_b.vd[12] + in_a.vd[16] * in_b.vd[19] - in_a.vd[17] * in_b.vd[18] + in_a.vd[18] * in_b.vd[17] - in_a.vd[19] * in_b.vd[16] - in_a.vd[20] * in_b.vd[23] + in_a.vd[21] * in_b.vd[22] - in_a.vd[22] * in_b.vd[21] + in_a.vd[23] * in_b.vd[20] - in_a.vd[24] * in_b.vd[27] + in_a.vd[25] * in_b.vd[26] - in_a.vd[26] * in_b.vd[25] + in_a.vd[27] * in_b.vd[24] + in_a.vd[28] * in_b.vd[31] - in_a.vd[29] * in_b.vd[30] + in_a.vd[30] * in_b.vd[29] - in_a.vd[31] * in_b.vd[28];
	out.vd[4] = in_a.vd[0] * in_b.vd[4] - in_a.vd[1] * in_b.vd[5] - in_a.vd[2] * in_b.vd[6] - in_a.vd[3] * in_b.vd[7] + in_a.vd[4] * in_b.vd[0] + in_a.vd[5] * in_b.vd[1] + in_a.vd[6] * in_b.vd[2] + in_a.vd[7] * in_b.vd[3] + in_a.vd[8] * in_b.vd[12] + in_a.vd[9] * in_b.vd[13] + in_a.vd[10] * in_b.vd[14] + in_a.vd[11] * in_b.vd[15] - in_a.vd[12] * in_b.vd[8] - in_a.vd[13] * in_b.vd[9] - in_a.vd[14] * in_b.vd[10] - in_a.vd[15] * in_b.vd[11] + in_a.vd[16] * in_b.vd[20] + in_a.vd[17] * in_b.vd[21] + in_a.vd[18] * in_b.vd[22] + in_a.vd[19] * in_b.vd[23] - in_a.vd[20] * in_b.vd[16] - in_a.vd[21] * in_b.vd[17] - in_a.vd[22] * in_b.vd[18] - in_a.vd[23] * in_b.vd[19] - in_a.vd[24] * in_b.vd[28] - in_a.vd[25] * in_b.vd[29] - in_a.vd[26] * in_b.vd[30] - in_a.vd[27] * in_b.vd[31] + in_a.vd[28] * in_b.vd[24] + in_a.vd[29] * in_b.vd[25] + in_a.vd[30] * in_b.vd[26] + in_a.vd[31] * in_b.vd[27];
	out.vd[5] = in_a.vd[0] * in_b.vd[5] + in_a.vd[1] * in_b.vd[4] - in_a.vd[2] * in_b.vd[7] + in_a.vd[3] * in_b.vd[6] - in_a.vd[4] * in_b.vd[1] + in_a.vd[5] * in_b.vd[0] - in_a.vd[6] * in_b.vd[3] + in_a.vd[7] * in_b.vd[2] + in_a.vd[8] * in_b.vd[13] - in_a.vd[9] * in_b.vd[12] + in_a.vd[10] * in_b.vd[15] - in_a.vd[11] * in_b.vd[14] + in_a.vd[12] * in_b.vd[9] - in_a.vd[13] * in_b.vd[8] + in_a.vd[14] * in_b.vd[11] - in_a.vd[15] * in_b.vd[10] + in_a.vd[16] * in_b.vd[21] - in_a.vd[17] * in_b.vd[20] + in_a.vd[18] * in_b.vd[23] - in_a.vd[19] * in_b.vd[22] + in_a.vd[20] * in_b.vd[17] - in_a.vd[21] * in_b.vd[16] + in_a.vd[22] * in_b.vd[19] - in_a.vd[23] * in_b.vd[18] - in_a.vd[24] * in_b.vd[29] + in_a.vd[25] * in_b.vd[28] - in_a.vd[26] * in_b.vd[31] + in_a.vd[27] * in_b.vd[30] - in_a.vd[28] * in_b.vd[25] + in_a.vd[29] * in_b.vd[24] - in_a.vd[30] * in_b.vd[27] + in_a.vd[31] * in_b.vd[26];
	out.vd[6] = in_a.vd[0] * in_b.vd[6] + in_a.vd[1] * in_b.vd[7] + in_a.vd[2] * in_b.vd[4] - in_a.vd[3] * in_b.vd[5] - in_a.vd[4] * in_b.vd[2] + in_a.vd[5] * in_b.vd[3] + in_a.vd[6] * in_b.vd[0] - in_a.vd[7] * in_b.vd[1] + in_a.vd[8] * in_b.vd[14] - in_a.vd[9] * in_b.vd[15] - in_a.vd[10] * in_b.vd[12] + in_a.vd[11] * in_b.vd[13] + in_a.vd[12] * in_b.vd[10] - in_a.vd[13] * in_b.vd[11] - in_a.vd[14] * in_b.vd[8] + in_a.vd[15] * in_b.vd[9] + in_a.vd[16] * in_b.vd[22] - in_a.vd[17] * in_b.vd[23] - in_a.vd[18] * in_b.vd[20] + in_a.vd[19] * in_b.vd[21] + in_a.vd[20] * in_b.vd[18] - in_a.vd[21] * in_b.vd[19] - in_a.vd[22] * in_b.vd[16] + in_a.vd[23] * in_b.vd[17] - in_a.vd[24] * in_b.vd[30] + in_a.vd[25] * in_b.vd[31] + in_a.vd[26] * in_b.vd[28] - in_a.vd[27] * in_b.vd[29] - in_a.vd[28] * in_b.vd[26] + in_a.vd[29] * in_b.vd[27] + in_a.vd[30] * in_b.vd[24] - in_a.vd[31] * in_b.vd[25];
	out.vd[7] = in_a.vd[0] * in_b.vd[7] - in_a.vd[1] * in_b.vd[6] + in_a.vd[2] * in_b.vd[5] + in_a.vd[3] * in_b.vd[4] - in_a.vd[4] * in_b.vd[3] - in_a.vd[5] * in_b.vd[2] + in_a.vd[6] * in_b.vd[1] + in_a.vd[7] * in_b.vd[0] + in_a.vd[8] * in_b.vd[15] + in_a.vd[9] * in_b.vd[14] - in_a.vd[10] * in_b.vd[13] - in_a.vd[11] * in_b.vd[12] + in_a.vd[12] * in_b.vd[11] + in_a.vd[13] * in_b.vd[10] - in_a.vd[14] * in_b.vd[9] - in_a.vd[15] * in_b.vd[8] + in_a.vd[16] * in_b.vd[23] + in_a.vd[17] * in_b.vd[22] - in_a.vd[18] * in_b.vd[21] - in_a.vd[19] * in_b.vd[20] + in_a.vd[20] * in_b.vd[19] + in_a.vd[21] * in_b.vd[18] - in_a.vd[22] * in_b.vd[17] - in_a.vd[23] * in_b.vd[16] - in_a.vd[24] * in_b.vd[31] - in_a.vd[25] * in_b.vd[30] + in_a.vd[26] * in_b.vd[29] + in_a.vd[27] * in_b.vd[28] - in_a.vd[28] * in_b.vd[27] - in_a.vd[29] * in_b.vd[26] + in_a.vd[30] * in_b.vd[25] + in_a.vd[31] * in_b.vd[24];
	out.vd[8] = in_a.vd[0] * in_b.vd[8] - in_a.vd[1] * in_b.vd[9] - in_a.vd[2] * in_b.vd[10] - in_a.vd[3] * in_b.vd[11] - in_a.vd[4] * in_b.vd[12] - in_a.vd[5] * in_b.vd[13] - in_a.vd[6] * in_b.vd[14] - in_a.vd[7] * in_b.vd[15] + in_a.vd[8] * in_b.vd[0] + in_a.vd[9] * in_b.vd[1] + in_a.vd[10] * in_b.vd[2] + in_a.vd[11] * in_b.vd[3] + in_a.vd[12] * in_b.vd[4] + in_a.vd[13] * in_b.vd[5] + in_a.vd[14] * in_b.vd[6] + in_a.vd[15] * in_b.vd[7] + in_a.vd[16] * in_b.vd[24] + in_a.vd[17] * in_b.vd[25] + in_a.vd[18] * in_b.vd[26] + in_a.vd[19] * in_b.vd[27] + in_a.vd[20] * in_b.vd[28] + in_a.vd[21] * in_b.vd[29] + in_a.vd[22] * in_b.vd[30] + in_a.vd[23] * in_b.vd[31] - in_a.vd[24] * in_b.vd[16] - in_a.vd[25] * in_b.vd[17] - in_a.vd[26] * in_b.vd[18] - in_a.vd[27] * in_b.vd[19] - in_a.vd[28] * in_b.vd[20] - in_a.vd[29] * in_b.vd[21] - in_a.vd[30] * in_b.vd[22] - in_a.vd[31] * in_b.vd[23];
	out.vd[9] = in_a.vd[0] * in_b.vd[9] + in_a.vd[1] * in_b.vd[8] - in_a.vd[2] * in_b.vd[11] + in_a.vd[3] * in_b.vd[10] - in_a.vd[4] * in_b.vd[13] + in_a.vd[5] * in_b.vd[12] + in_a.vd[6] * in_b.vd[15] - in_a.vd[7] * in_b.vd[14] - in_a.vd[8] * in_b.vd[1] + in_a.vd[9] * in_b.vd[0] - in_a.vd[10] * in_b.vd[3] + in_a.vd[11] * in_b.vd[2] - in_a.vd[12] * in_b.vd[5] + in_a.vd[13] * in_b.vd[4] + in_a.vd[14] * in_b.vd[7] - in_a.vd[15] * in_b.vd[6] + in_a.vd[16] * in_b.vd[25] - in_a.vd[17] * in_b.vd[24] + in_a.vd[18] * in_b.vd[27] - in_a.vd[19] * in_b.vd[26] + in_a.vd[20] * in_b.vd[29] - in_a.vd[21] * in_b.vd[28] - in_a.vd[22] * in_b.vd[31] + in_a.vd[23] * in_b.vd[30] + in_a.vd[24] * in_b.vd[17] - in_a.vd[25] * in_b.vd[16] + in_a.vd[26] * in_b.vd[19] - in_a.vd[27] * in_b.vd[18] + in_a.vd[28] * in_b.vd[21] - in_a.vd[29] * in_b.vd[20] - in_a.vd[30] * in_b.vd[23] + in_a.vd[31] * in_b.vd[22];
	out.vd[10] = in_a.vd[0] * in_b.vd[10] + in_a.vd[1] * in_b.vd[11] + in_a.vd[2] * in_b.vd[8] - in_a.vd[3] * in_b.vd[9] - in_a.vd[4] * in_b.vd[14] - in_a.vd[5] * in_b.vd[15] + in_a.vd[6] * in_b.vd[12] + in_a.vd[7] * in_b.vd[13] - in_a.vd[8] * in_b.vd[2] + in_a.vd[9] * in_b.vd[3] + in_a.vd[10] * in_b.vd[0] - in_a.vd[11] * in_b.vd[1] - in_a.vd[12] * in_b.vd[6] - in_a.vd[13] * in_b.vd[7] + in_a.vd[14] * in_b.vd[4] + in_a.vd[15] * in_b.vd[5] + in_a.vd[16] * in_b.vd[26] - in_a.vd[17] * in_b.vd[27] - in_a.vd[18] * in_b.vd[24] + in_a.vd[19] * in_b.vd[25] + in_a.vd[20] * in_b.vd[30] + in_a.vd[21] * in_b.vd[31] - in_a.vd[22] * in_b.vd[28] - in_a.vd[23] * in_b.vd[29] + in_a.vd[24] * in_b.vd[18] - in_a.vd[25] * in_b.vd[19] - in_a.vd[26] * in_b.vd[16] + in_a.vd[27] * in_b.vd[17] + in_a.vd[28] * in_b.vd[22] + in_a.vd[29] * in_b.vd[23] - in_a.vd[30] * in_b.vd[20] - in_a.vd[31] * in_b.vd[21];
	out.vd[11] = in_a.vd[0] * in_b.vd[11] - in_a.vd[1] * in_b.vd[10] + in_a.vd[2] * in_b.vd[9] + in_a.vd[3] * in_b.vd[8] - in_a.vd[4] * in_b.vd[15] + in_a.vd[5] * in_b.vd[14] - in_a.vd[6] * in_b.vd[13] + in_a.vd[7] * in_b.vd[12] - in_a.vd[8] * in_b.vd[3] - in_a.vd[9] * in_b.vd[2] + in_a.vd[10] * in_b.vd[1] + in_a.vd[11] * in_b.vd[0] - in_a.vd[12] * in_b.vd[7] + in_a.vd[13] * in_b.vd[6] - in_a.vd[14] * in_b.vd[5] + in_a.vd[15] * in_b.vd[4] + in_a.vd[16] * in_b.vd[27] + in_a.vd[17] * in_b.vd[26] - in_a.vd[18] * in_b.vd[25] - in_a.vd[19] * in_b.vd[24] + in_a.vd[20] * in_b.vd[31] - in_a.vd[21] * in_b.vd[30] + in_a.vd[22] * in_b.vd[29] - in_a.vd[23] * in_b.vd[28] + in_a.vd[24] * in_b.vd[19] + in_a.vd[25] * in_b.vd[18] - in_a.vd[26] * in_b.vd[17] - in_a.vd[27] * in_b.vd[16] + in_a.vd[28] * in_b.vd[23] - in_a.vd[29] * in_b.vd[22] + in_a.vd[30] * in_b.vd[21] - in_a.vd[31] * in_b.vd[20];
	out.vd[12] = in_a.vd[0] * in_b.vd[12] + in_a.vd[1] * in_b.vd[13] + in_a.vd[2] * in_b.vd[14] + in_a.vd[3] * in_b.vd[15] + in_a.vd[4] * in_b.vd[8] - in_a.vd[5] * in_b.vd[9] - in_a.vd[6] * in_b.vd[10] - in_a.vd[7] * in_b.vd[11] - in_a.vd[8] * in_b.vd[4] + in_a.vd[9] * in_b.vd[5] + in_a.vd[10] * in_b.vd[6] + in_a.vd[11] * in_b.vd[7] + in_a.vd[12] * in_b.vd[0] - in_a.vd[13] * in_b.vd[1] - in_a.vd[14] * in_b.vd[2] - in_a.vd[15] * in_b.vd[3] + in_a.vd[16] * in_b.vd[28] - in_a.vd[17] * in_b.vd[29] - in_a.vd[18] * in_b.vd[30] - in_a.vd[19] * in_b.vd[31] - in_a.vd[20] * in_b.vd[24] + in_a.vd[21] * in_b.vd[25] + in_a.vd[22] * in_b.vd[26] + in_a.vd[23] * in_b.vd[27] + in_a.vd[24] * in_b.vd[20] - in_a.vd[25] * in_b.vd[21] - in_a.vd[26] * in_b.vd[22] - in_a.vd[27] * in_b.vd[23] - in_a.vd[28] * in_b.vd[16] + in_a.vd[29] * in_b.vd[17] + in_a.vd[30] * in_b.vd[18] + in_a.vd[31] * in_b.vd[19];
	out.vd[13] = in_a.vd[0] * in_b.vd[13] - in_a.vd[1] * in_b.vd[12] + in_a.vd[2] * in_b.vd[15] - in_a.vd[3] * in_b.vd[14] + in_a.vd[4] * in_b.vd[9] + in_a.vd[5] * in_b.vd[8] + in_a.vd[6] * in_b.vd[11] - in_a.vd[7] * in_b.vd[10] - in_a.vd[8] * in_b.vd[5] - in_a.vd[9] * in_b.vd[4] + in_a.vd[10] * in_b.vd[7] - in_a.vd[11] * in_b.vd[6] + in_a.vd[12] * in_b.vd[1] + in_a.vd[13] * in_b.vd[0] + in_a.vd[14] * in_b.vd[3] - in_a.vd[15] * in_b.vd[2] + in_a.vd[16] * in_b.vd[29] + in_a.vd[17] * in_b.vd[28] - in_a.vd[18] * in_b.vd[31] + in_a.vd[19] * in_b.vd[30] - in_a.vd[20] * in_b.vd[25] - in_a.vd[21] * in_b.vd[24] - in_a.vd[22] * in_b.vd[27] + in_a.vd[23] * in_b.vd[26] + in_a.vd[24] * in_b.vd[21] + in_a.vd[25] * in_b.vd[20] - in_a.vd[26] * in_b.vd[23] + in_a.vd[27] * in_b.vd[22] - in_a.vd[28] * in_b.vd[17] - in_a.vd[29] * in_b.vd[16] - in_a.vd[30] * in_b.vd[19] + in_a.vd[31] * in_b.vd[18];
	out.vd[14] = in_a.vd[0] * in_b.vd[14] - in_a.vd[1] * in_b.vd[15] - in_a.vd[2] * in_b.vd[12] + in_a.vd[3] * in_b.vd[13] + in_a.vd[4] * in_b.vd[10] - in_a.vd[5] * in_b.vd[11] + in_a.vd[6] * in_b.vd[8] + in_a.vd[7] * in_b.vd[9] - in_a.vd[8] * in_b.vd[6] - in_a.vd[9] * in_b.vd[7] - in_a.vd[10] * in_b.vd[4] + in_a.vd[11] * in_b.vd[5] + in_a.vd[12] * in_b.vd[2] - in_a.vd[13] * in_b.vd[3] + in_a.vd[14] * in_b.vd[0] + in_a.vd[15] * in_b.vd[1] + in_a.vd[16] * in_b.vd[30] + in_a.vd[17] * in_b.vd[31] + in_a.vd[18] * in_b.vd[28] - in_a.vd[19] * in_b.vd[29] - in_a.vd[20] * in_b.vd[26] + in_a.vd[21] * in_b.vd[27] - in_a.vd[22] * in_b.vd[24] - in_a.vd[23] * in_b.vd[25] + in_a.vd[24] * in_b.vd[22] + in_a.vd[25] * in_b.vd[23] + in_a.vd[26] * in_b.vd[20] - in_a.vd[27] * in_b.vd[21] - in_a.vd[28] * in_b.vd[18] + in_a.vd[29] * in_b.vd[19] - in_a.vd[30] * in_b.vd[16] - in_a.vd[31] * in_b.vd[17];
	out.vd[15] = in_a.vd[0] * in_b.vd[15] + in_a.vd[1] * in_b.vd[14] - in_a.vd[2] * in_b.vd[13] - in_a.vd[3] * in_b.vd[12] + in_a.vd[4] * in_b.vd[11] + in_a.vd[5] * in_b.vd[10] - in_a.vd[6] * in_b.vd[9] + in_a.vd[7] * in_b.vd[8] - in_a.vd[8] * in_b.vd[7] + in_a.vd[9] * in_b.vd[6] - in_a.vd[10] * in_b.vd[5] - in_a.vd[11] * in_b.vd[4] + in_a.vd[12] * in_b.vd[3] + in_a.vd[13] * in_b.vd[2] - in_a.vd[14] * in_b.vd[1] + in_a.vd[15] * in_b.vd[0] + in_a.vd[16] * in_b.vd[31] - in_a.vd[17] * in_b.vd[30] + in_a.vd[18] * in_b.vd[29] + in_a.vd[19] * in_b.vd[28] - in_a.vd[20] * in_b.vd[27] - in_a.vd[21] * in_b.vd[26] + in_a.vd[22] * in_b.vd[25] - in_a.vd[23] * in_b.vd[24] + in_a.vd[24] * in_b.vd[23] - in_a.vd[25] * in_b.vd[22] + in_a.vd[26] * in_b.vd[21] + in_a.vd[27] * in_b.vd[20] - in_a.vd[28] * in_b.vd[19] - in_a.vd[29] * in_b.vd[18] + in_a.vd[30] * in_b.vd[17] - in_a.vd[31] * in_b.vd[16];
	out.vd[16] = in_a.vd[0] * in_b.vd[16] - in_a.vd[1] * in_b.vd[17] - in_a.vd[2] * in_b.vd[18] - in_a.vd[3] * in_b.vd[19] - in_a.vd[4] * in_b.vd[20] - in_a.vd[5] * in_b.vd[21] - in_a.vd[6] * in_b.vd[22] - in_a.vd[7] * in_b.vd[23] - in_a.vd[8] * in_b.vd[24] - in_a.vd[9] * in_b.vd[25] - in_a.vd[10] * in_b.vd[26] - in_a.vd[11] * in_b.vd[27] - in_a.vd[12] * in_b.vd[28] - in_a.vd[13] * in_b.vd[29] - in_a.vd[14] * in_b.vd[30] - in_a.vd[15] * in_b.vd[31] + in_a.vd[16] * in_b.vd[0] + in_a.vd[17] * in_b.vd[1] + in_a.vd[18] * in_b.vd[2] + in_a.vd[19] * in_b.vd[3] + in_a.vd[20] * in_b.vd[4] + in_a.vd[21] * in_b.vd[5] + in_a.vd[22] * in_b.vd[6] + in_a.vd[23] * in_b.vd[7] + in_a.vd[24] * in_b.vd[8] + in_a.vd[25] * in_b.vd[9] + in_a.vd[26] * in_b.vd[10] + in_a.vd[27] * in_b.vd[11] + in_a.vd[28] * in_b.vd[12] + in_a.vd[29] * in_b.vd[13] + in_a.vd[30] * in_b.vd[14] + in_a.vd[31] * in_b.vd[15];
	out.vd[17] = in_a.vd[0] * in_b.vd[17] + in_a.vd[1] * in_b.vd[16] - in_a.vd[2] * in_b.vd[19] + in_a.vd[3] * in_b.vd[18] - in_a.vd[4] * in_b.vd[21] + in_a.vd[5] * in_b.vd[20] + in_a.vd[6] * in_b.vd[23] - in_a.vd[7] * in_b.vd[22] - in_a.vd[8] * in_b.vd[25] + in_a.vd[9] * in_b.vd[24] + in_a.vd[10] * in_b.vd[27] - in_a.vd[11] * in_b.vd[26] + in_a.vd[12] * in_b.vd[29] - in_a.vd[13] * in_b.vd[28] - in_a.vd[14] * in_b.vd[31] + in_a.vd[15] * in_b.vd[30] - in_a.vd[16] * in_b.vd[1] + in_a.vd[17] * in_b.vd[0] - in_a.vd[18] * in_b.vd[3] + in_a.vd[19] * in_b.vd[2] - in_a.vd[20] * in_b.vd[5] + in_a.vd[21] * in_b.vd[4] + in_a.vd[22] * in_b.vd[7] - in_a.vd[23] * in_b.vd[6] - in_a.vd[24] * in_b.vd[9] + in_a.vd[25] * in_b.vd[8] + in_a.vd[26] * in_b.vd[11] - in_a.vd[27] * in_b.vd[10] + in_a.vd[28] * in_b.vd[13] - in_a.vd[29] * in_b.vd[12] - in_a.vd[30] * in_b.vd[15] + in_a.vd[31] * in_b.vd[14];
	out.vd[18] = in_a.vd[0] * in_b.vd[18] + in_a.vd[1] * in_b.vd[19] + in_a.vd[2] * in_b.vd[16] - in_a.vd[3] * in_b.vd[17] - in_a.vd[4] * in_b.vd[22] - in_a.vd[5] * in_b.vd[23] + in_a.vd[6] * in_b.vd[20] + in_a.vd[7] * in_b.vd[21] - in_a.vd[8] * in_b.vd[26] - in_a.vd[9] * in_b.vd[27] + in_a.vd[10] * in_b.vd[24] + in_a.vd[11] * in_b.vd[25] + in_a.vd[12] * in_b.vd[30] + in_a.vd[13] * in_b.vd[31] - in_a.vd[14] * in_b.vd[28] - in_a.vd[15] * in_b.vd[29] - in_a.vd[16] * in_b.vd[2] + in_a.vd[17] * in_b.vd[3] + in_a.vd[18] * in_b.vd[0] - in_a.vd[19] * in_b.vd[1] - in_a.vd[20] * in_b.vd[6] - in_a.vd[21] * in_b.vd[7] + in_a.vd[22] * in_b.vd[4] + in_a.vd[23] * in_b.vd[5] - in_a.vd[24] * in_b.vd[10] - in_a.vd[25] * in_b.vd[11] + in_a.vd[26] * in_b.vd[8] + in_a.vd[27] * in_b.vd[9] + in_a.vd[28] * in_b.vd[14] + in_a.vd[29] * in_b.vd[15] - in_a.vd[30] * in_b.vd[12] - in_a.vd[31] * in_b.vd[13];
	out.vd[19] = in_a.vd[0] * in_b.vd[19] - in_a.vd[1] * in_b.vd[18] + in_a.vd[2] * in_b.vd[17] + in_a.vd[3] * in_b.vd[16] - in_a.vd[4] * in_b.vd[23] + in_a.vd[5] * in_b.vd[22] - in_a.vd[6] * in_b.vd[21] + in_a.vd[7] * in_b.vd[20] - in_a.vd[8] * in_b.vd[27] + in_a.vd[9] * in_b.vd[26] - in_a.vd[10] * in_b.vd[25] + in_a.vd[11] * in_b.vd[24] + in_a.vd[12] * in_b.vd[31] - in_a.vd[13] * in_b.vd[30] + in_a.vd[14] * in_b.vd[29] - in_a.vd[15] * in_b.vd[28] - in_a.vd[16] * in_b.vd[3] - in_a.vd[17] * in_b.vd[2] + in_a.vd[18] * in_b.vd[1] + in_a.vd[19] * in_b.vd[0] - in_a.vd[20] * in_b.vd[7] + in_a.vd[21] * in_b.vd[6] - in_a.vd[22] * in_b.vd[5] + in_a.vd[23] * in_b.vd[4] - in_a.vd[24] * in_b.vd[11] + in_a.vd[25] * in_b.vd[10] - in_a.vd[26] * in_b.vd[9] + in_a.vd[27] * in_b.vd[8] + in_a.vd[28] * in_b.vd[15] - in_a.vd[29] * in_b.vd[14] + in_a.vd[30] * in_b.vd[13] - in_a.vd[31] * in_b.vd[12];
	out.vd[20] = in_a.vd[0] * in_b.vd[20] + in_a.vd[1] * in_b.vd[21] + in_a.vd[2] * in_b.vd[22] + in_a.vd[3] * in_b.vd[23] + in_a.vd[4] * in_b.vd[16] - in_a.vd[5] * in_b.vd[17] - in_a.vd[6] * in_b.vd[18] - in_a.vd[7] * in_b.vd[19] - in_a.vd[8] * in_b.vd[28] - in_a.vd[9] * in_b.vd[29] - in_a.vd[10] * in_b.vd[30] - in_a.vd[11] * in_b.vd[31] + in_a.vd[12] * in_b.vd[24] + in_a.vd[13] * in_b.vd[25] + in_a.vd[14] * in_b.vd[26] + in_a.vd[15] * in_b.vd[27] - in_a.vd[16] * in_b.vd[4] + in_a.vd[17] * in_b.vd[5] + in_a.vd[18] * in_b.vd[6] + in_a.vd[19] * in_b.vd[7] + in_a.vd[20] * in_b.vd[0] - in_a.vd[21] * in_b.vd[1] - in_a.vd[22] * in_b.vd[2] - in_a.vd[23] * in_b.vd[3] - in_a.vd[24] * in_b.vd[12] - in_a.vd[25] * in_b.vd[13] - in_a.vd[26] * in_b.vd[14] - in_a.vd[27] * in_b.vd[15] + in_a.vd[28] * in_b.vd[8] + in_a.vd[29] * in_b.vd[9] + in_a.vd[30] * in_b.vd[10] + in_a.vd[31] * in_b.vd[11];
	out.vd[21] = in_a.vd[0] * in_b.vd[21] - in_a.vd[1] * in_b.vd[20] + in_a.vd[2] * in_b.vd[23] - in_a.vd[3] * in_b.vd[22] + in_a.vd[4] * in_b.vd[17] + in_a.vd[5] * in_b.vd[16] + in_a.vd[6] * in_b.vd[19] - in_a.vd[7] * in_b.vd[18] - in_a.vd[8] * in_b.vd[29] + in_a.vd[9] * in_b.vd[28] - in_a.vd[10] * in_b.vd[31] + in_a.vd[11] * in_b.vd[30] - in_a.vd[12] * in_b.vd[25] + in_a.vd[13] * in_b.vd[24] - in_a.vd[14] * in_b.vd[27] + in_a.vd[15] * in_b.vd[26] - in_a.vd[16] * in_b.vd[5] - in_a.vd[17] * in_b.vd[4] + in_a.vd[18] * in_b.vd[7] - in_a.vd[19] * in_b.vd[6] + in_a.vd[20] * in_b.vd[1] + in_a.vd[21] * in_b.vd[0] + in_a.vd[22] * in_b.vd[3] - in_a.vd[23] * in_b.vd[2] - in_a.vd[24] * in_b.vd[13] + in_a.vd[25] * in_b.vd[12] - in_a.vd[26] * in_b.vd[15] + in_a.vd[27] * in_b.vd[14] - in_a.vd[28] * in_b.vd[9] + in_a.vd[29] * in_b.vd[8] - in_a.vd[30] * in_b.vd[11] + in_a.vd[31] * in_b.vd[10];
	out.vd[22] = in_a.vd[0] * in_b.vd[22] - in_a.vd[1] * in_b.vd[23] - in_a.vd[2] * in_b.vd[20] + in_a.vd[3] * in_b.vd[21] + in_a.vd[4] * in_b.vd[18] - in_a.vd[5] * in_b.vd[19] + in_a.vd[6] * in_b.vd[16] + in_a.vd[7] * in_b.vd[17] - in_a.vd[8] * in_b.vd[30] + in_a.vd[9] * in_b.vd[31] + in_a.vd[10] * in_b.vd[28] - in_a.vd[11] * in_b.vd[29] - in_a.vd[12] * in_b.vd[26] + in_a.vd[13] * in_b.vd[27] + in_a.vd[14] * in_b.vd[24] - in_a.vd[15] * in_b.vd[25] - in_a.vd[16] * in_b.vd[6] - in_a.vd[17] * in_b.vd[7] - in_a.vd[18] * in_b.vd[4] + in_a.vd[19] * in_b.vd[5] + in_a.vd[20] * in_b.vd[2] - in_a.vd[21] * in_b.vd[3] + in_a.vd[22] * in_b.vd[0] + in_a.vd[23] * in_b.vd[1] - in_a.vd[24] * in_b.vd[14] + in_a.vd[25] * in_b.vd[15] + in_a.vd[26] * in_b.vd[12] - in_a.vd[27] * in_b.vd[13] - in_a.vd[28] * in_b.vd[10] + in_a.vd[29] * in_b.vd[11] + in_a.vd[30] * in_b.vd[8] - in_a.vd[31] * in_b.vd[9];
	out.vd[23] = in_a.vd[0] * in_b.vd[23] + in_a.vd[1] * in_b.vd[22] - in_a.vd[2] * in_b.vd[21] - in_a.vd[3] * in_b.vd[20] + in_a.vd[4] * in_b.vd[19] + in_a.vd[5] * in_b.vd[18] - in_a.vd[6] * in_b.vd[17] + in_a.vd[7] * in_b.vd[16] - in_a.vd[8] * in_b.vd[31] - in_a.vd[9] * in_b.vd[30] + in_a.vd[10] * in_b.vd[29] + in_a.vd[11] * in_b.vd[28] - in_a.vd[12] * in_b.vd[27] - in_a.vd[13] * in_b.vd[26] + in_a.vd[14] * in_b.vd[25] + in_a.vd[15] * in_b.vd[24] - in_a.vd[16] * in_b.vd[7] + in_a.vd[17] * in_b.vd[6] - in_a.vd[18] * in_b.vd[5] - in_a.vd[19] * in_b.vd[4] + in_a.vd[20] * in_b.vd[3] + in_a.vd[21] * in_b.vd[2] - in_a.vd[22] * in_b.vd[1] + in_a.vd[23] * in_b.vd[0] - in_a.vd[24] * in_b.vd[15] - in_a.vd[25] * in_b.vd[14] + in_a.vd[26] * in_b.vd[13] + in_a.vd[27] * in_b.vd[12] - in_a.vd[28] * in_b.vd[11] - in_a.vd[29] * in_b.vd[10] + in_a.vd[30] * in_b.vd[9] + in_a.vd[31] * in_b.vd[8];
	out.vd[24] = in_a.vd[0] * in_b.vd[24] + in_a.vd[1] * in_b.vd[25] + in_a.vd[2] * in_b.vd[26] + in_a.vd[3] * in_b.vd[27] + in_a.vd[4] * in_b.vd[28] + in_a.vd[5] * in_b.vd[29] + in_a.vd[6] * in_b.vd[30] + in_a.vd[7] * in_b.vd[31] + in_a.vd[8] * in_b.vd[16] - in_a.vd[9] * in_b.vd[17] - in_a.vd[10] * in_b.vd[18] - in_a.vd[11] * in_b.vd[19] - in_a.vd[12] * in_b.vd[20] - in_a.vd[13] * in_b.vd[21] - in_a.vd[14] * in_b.vd[22] - in_a.vd[15] * in_b.vd[23] - in_a.vd[16] * in_b.vd[8] + in_a.vd[17] * in_b.vd[9] + in_a.vd[18] * in_b.vd[10] + in_a.vd[19] * in_b.vd[11] + in_a.vd[20] * in_b.vd[12] + in_a.vd[21] * in_b.vd[13] + in_a.vd[22] * in_b.vd[14] + in_a.vd[23] * in_b.vd[15] + in_a.vd[24] * in_b.vd[0] - in_a.vd[25] * in_b.vd[1] - in_a.vd[26] * in_b.vd[2] - in_a.vd[27] * in_b.vd[3] - in_a.vd[28] * in_b.vd[4] - in_a.vd[29] * in_b.vd[5] - in_a.vd[30] * in_b.vd[6] - in_a.vd[31] * in_b.vd[7];
	out.vd[25] = in_a.vd[0] * in_b.vd[25] - in_a.vd[1] * in_b.vd[24] + in_a.vd[2] * in_b.vd[27] - in_a.vd[3] * in_b.vd[26] + in_a.vd[4] * in_b.vd[29] - in_a.vd[5] * in_b.vd[28] - in_a.vd[6] * in_b.vd[31] + in_a.vd[7] * in_b.vd[30] + in_a.vd[8] * in_b.vd[17] + in_a.vd[9] * in_b.vd[16] + in_a.vd[10] * in_b.vd[19] - in_a.vd[11] * in_b.vd[18] + in_a.vd[12] * in_b.vd[21] - in_a.vd[13] * in_b.vd[20] - in_a.vd[14] * in_b.vd[23] + in_a.vd[15] * in_b.vd[22] - in_a.vd[16] * in_b.vd[9] - in_a.vd[17] * in_b.vd[8] + in_a.vd[18] * in_b.vd[11] - in_a.vd[19] * in_b.vd[10] + in_a.vd[20] * in_b.vd[13] - in_a.vd[21] * in_b.vd[12] - in_a.vd[22] * in_b.vd[15] + in_a.vd[23] * in_b.vd[14] + in_a.vd[24] * in_b.vd[1] + in_a.vd[25] * in_b.vd[0] + in_a.vd[26] * in_b.vd[3] - in_a.vd[27] * in_b.vd[2] + in_a.vd[28] * in_b.vd[5] - in_a.vd[29] * in_b.vd[4] - in_a.vd[30] * in_b.vd[7] + in_a.vd[31] * in_b.vd[6];
	out.vd[26] = in_a.vd[0] * in_b.vd[26] - in_a.vd[1] * in_b.vd[27] - in_a.vd[2] * in_b.vd[24] + in_a.vd[3] * in_b.vd[25] + in_a.vd[4] * in_b.vd[30] + in_a.vd[5] * in_b.vd[31] - in_a.vd[6] * in_b.vd[28] - in_a.vd[7] * in_b.vd[29] + in_a.vd[8] * in_b.vd[18] - in_a.vd[9] * in_b.vd[19] + in_a.vd[10] * in_b.vd[16] + in_a.vd[11] * in_b.vd[17] + in_a.vd[12] * in_b.vd[22] + in_a.vd[13] * in_b.vd[23] - in_a.vd[14] * in_b.vd[20] - in_a.vd[15] * in_b.vd[21] - in_a.vd[16] * in_b.vd[10] - in_a.vd[17] * in_b.vd[11] - in_a.vd[18] * in_b.vd[8] + in_a.vd[19] * in_b.vd[9] + in_a.vd[20] * in_b.vd[14] + in_a.vd[21] * in_b.vd[15] - in_a.vd[22] * in_b.vd[12] - in_a.vd[23] * in_b.vd[13] + in_a.vd[24] * in_b.vd[2] - in_a.vd[25] * in_b.vd[3] + in_a.vd[26] * in_b.vd[0] + in_a.vd[27] * in_b.vd[1] + in_a.vd[28] * in_b.vd[6] + in_a.vd[29] * in_b.vd[7] - in_a.vd[30] * in_b.vd[4] - in_a.vd[31] * in_b.vd[5];
	out.vd[27] = in_a.vd[0] * in_b.vd[27] + in_a.vd[1] * in_b.vd[26] - in_a.vd[2] * in_b.vd[25] - in_a.vd[3] * in_b.vd[24] + in_a.vd[4] * in_b.vd[31] - in_a.vd[5] * in_b.vd[30] + in_a.vd[6] * in_b.vd[29] - in_a.vd[7] * in_b.vd[28] + in_a.vd[8] * in_b.vd[19] + in_a.vd[9] * in_b.vd[18] - in_a.vd[10] * in_b.vd[17] + in_a.vd[11] * in_b.vd[16] + in_a.vd[12] * in_b.vd[23] - in_a.vd[13] * in_b.vd[22] + in_a.vd[14] * in_b.vd[21] - in_a.vd[15] * in_b.vd[20] - in_a.vd[16] * in_b.vd[11] + in_a.vd[17] * in_b.vd[10] - in_a.vd[18] * in_b.vd[9] - in_a.vd[19] * in_b.vd[8] + in_a.vd[20] * in_b.vd[15] - in_a.vd[21] * in_b.vd[14] + in_a.vd[22] * in_b.vd[13] - in_a.vd[23] * in_b.vd[12] + in_a.vd[24] * in_b.vd[3] + in_a.vd[25] * in_b.vd[2] - in_a.vd[26] * in_b.vd[1] + in_a.vd[27] * in_b.vd[0] + in_a.vd[28] * in_b.vd[7] - in_a.vd[29] * in_b.vd[6] + in_a.vd[30] * in_b.vd[5] - in_a.vd[31] * in_b.vd[4];
	out.vd[28] = in_a.vd[0] * in_b.vd[28] - in_a.vd[1] * in_b.vd[29] - in_a.vd[2] * in_b.vd[30] - in_a.vd[3] * in_b.vd[31] - in_a.vd[4] * in_b.vd[24] + in_a.vd[5] * in_b.vd[25] + in_a.vd[6] * in_b.vd[26] + in_a.vd[7] * in_b.vd[27] + in_a.vd[8] * in_b.vd[20] - in_a.vd[9] * in_b.vd[21] - in_a.vd[10] * in_b.vd[22] - in_a.vd[11] * in_b.vd[23] + in_a.vd[12] * in_b.vd[16] + in_a.vd[13] * in_b.vd[17] + in_a.vd[14] * in_b.vd[18] + in_a.vd[15] * in_b.vd[19] - in_a.vd[16] * in_b.vd[12] - in_a.vd[17] * in_b.vd[13] - in_a.vd[18] * in_b.vd[14] - in_a.vd[19] * in_b.vd[15] - in_a.vd[20] * in_b.vd[8] + in_a.vd[21] * in_b.vd[9] + in_a.vd[22] * in_b.vd[10] + in_a.vd[23] * in_b.vd[11] + in_a.vd[24] * in_b.vd[4] - in_a.vd[25] * in_b.vd[5] - in_a.vd[26] * in_b.vd[6] - in_a.vd[27] * in_b.vd[7] + in_a.vd[28] * in_b.vd[0] + in_a.vd[29] * in_b.vd[1] + in_a.vd[30] * in_b.vd[2] + in_a.vd[31] * in_b.vd[3];
	out.vd[29] = in_a.vd[0] * in_b.vd[29] + in_a.vd[1] * in_b.vd[28] - in_a.vd[2] * in_b.vd[31] + in_a.vd[3] * in_b.vd[30] - in_a.vd[4] * in_b.vd[25] - in_a.vd[5] * in_b.vd[24] - in_a.vd[6] * in_b.vd[27] + in_a.vd[7] * in_b.vd[26] + in_a.vd[8] * in_b.vd[21] + in_a.vd[9] * in_b.vd[20] - in_a.vd[10] * in_b.vd[23] + in_a.vd[11] * in_b.vd[22] - in_a.vd[12] * in_b.vd[17] + in_a.vd[13] * in_b.vd[16] - in_a.vd[14] * in_b.vd[19] + in_a.vd[15] * in_b.vd[18] - in_a.vd[16] * in_b.vd[13] + in_a.vd[17] * in_b.vd[12] - in_a.vd[18] * in_b.vd[15] + in_a.vd[19] * in_b.vd[14] - in_a.vd[20] * in_b.vd[9] - in_a.vd[21] * in_b.vd[8] - in_a.vd[22] * in_b.vd[11] + in_a.vd[23] * in_b.vd[10] + in_a.vd[24] * in_b.vd[5] + in_a.vd[25] * in_b.vd[4] - in_a.vd[26] * in_b.vd[7] + in_a.vd[27] * in_b.vd[6] - in_a.vd[28] * in_b.vd[1] + in_a.vd[29] * in_b.vd[0] - in_a.vd[30] * in_b.vd[3] + in_a.vd[31] * in_b.vd[2];
	out.vd[30] = in_a.vd[0] * in_b.vd[30] + in_a.vd[1] * in_b.vd[31] + in_a.vd[2] * in_b.vd[28] - in_a.vd[3] * in_b.vd[29] - in_a.vd[4] * in_b.vd[26] + in_a.vd[5] * in_b.vd[27] - in_a.vd[6] * in_b.vd[24] - in_a.vd[7] * in_b.vd[25] + in_a.vd[8] * in_b.vd[22] + in_a.vd[9] * in_b.vd[23] + in_a.vd[10] * in_b.vd[20] - in_a.vd[11] * in_b.vd[21] - in_a.vd[12] * in_b.vd[18] + in_a.vd[13] * in_b.vd[19] + in_a.vd[14] * in_b.vd[16] - in_a.vd[15] * in_b.vd[17] - in_a.vd[16] * in_b.vd[14] + in_a.vd[17] * in_b.vd[15] + in_a.vd[18] * in_b.vd[12] - in_a.vd[19] * in_b.vd[13] - in_a.vd[20] * in_b.vd[10] + in_a.vd[21] * in_b.vd[11] - in_a.vd[22] * in_b.vd[8] - in_a.vd[23] * in_b.vd[9] + in_a.vd[24] * in_b.vd[6] + in_a.vd[25] * in_b.vd[7] + in_a.vd[26] * in_b.vd[4] - in_a.vd[27] * in_b.vd[5] - in_a.vd[28] * in_b.vd[2] + in_a.vd[29] * in_b.vd[3] + in_a.vd[30] * in_b.vd[0] - in_a.vd[31] * in_b.vd[1];
	out.vd[31] = in_a.vd[0] * in_b.vd[31] - in_a.vd[1] * in_b.vd[30] + in_a.vd[2] * in_b.vd[29] + in_a.vd[3] * in_b.vd[28] - in_a.vd[4] * in_b.vd[27] - in_a.vd[5] * in_b.vd[26] + in_a.vd[6] * in_b.vd[25] - in_a.vd[7] * in_b.vd[24] + in_a.vd[8] * in_b.vd[23] - in_a.vd[9] * in_b.vd[22] + in_a.vd[10] * in_b.vd[21] + in_a.vd[11] * in_b.vd[20] - in_a.vd[12] * in_b.vd[19] - in_a.vd[13] * in_b.vd[18] + in_a.vd[14] * in_b.vd[17] + in_a.vd[15] * in_b.vd[16] - in_a.vd[16] * in_b.vd[15] - in_a.vd[17] * in_b.vd[14] + in_a.vd[18] * in_b.vd[13] + in_a.vd[19] * in_b.vd[12] - in_a.vd[20] * in_b.vd[11] - in_a.vd[21] * in_b.vd[10] + in_a.vd[22] * in_b.vd[9] - in_a.vd[23] * in_b.vd[8] + in_a.vd[24] * in_b.vd[7] - in_a.vd[25] * in_b.vd[6] + in_a.vd[26] * in_b.vd[5] + in_a.vd[27] * in_b.vd[4] - in_a.vd[28] * in_b.vd[3] - in_a.vd[29] * in_b.vd[2] + in_a.vd[30] * in_b.vd[1] + in_a.vd[31] * in_b.vd[0];

	return out;
}

// Exponential function (base = e) for variable T and N
template<class T, size_t N>
vertex<T, N> exp(const vertex<T, N>& in)
{
	T all_self_dot = 0;
	T imag_self_dot = 0;
	vertex<T, N> out;

	for (size_t i = 1; i < N; i++)
		imag_self_dot += (in.vd[i] * in.vd[i]);

	all_self_dot = imag_self_dot + (in.vd[0] * in.vd[0]);

	if (all_self_dot == 0)
	{
		for (size_t i = 0; i < N; i++)
			out.vd[i] = 0;

		return out;
	}

	//	const T l_d = sqrt(all_self_dot); // not needed
	const T l_e = sqrt(imag_self_dot);

	out.vd[0] = exp(in.vd[0]) * cos(l_e);

	if (l_e != 0)
	{
		const T x = 1 / l_e;
		const T y = exp(in.vd[0]) * sin(l_e);
		const T z = x * y;

		for (size_t i = 1; i < N; i++)
			out.vd[i] = in.vd[i] * z;
	}

	return out;
}

// Log function (base = e) for variable T and N
template<class T, size_t N>
vertex<T, N> log(const vertex<T, N>& in)
{
	T all_self_dot = 0;
	T imag_self_dot = 0;
	vertex<T, N> out;

	for (size_t i = 1; i < N; i++)
		imag_self_dot += (in.vd[i] * in.vd[i]);

	all_self_dot = imag_self_dot + (in.vd[0] * in.vd[0]);

	if (all_self_dot == 0)
	{
		for (size_t i = 0; i < N; i++)
			out.vd[i] = 0;

		return out;
	}

	const T l_d = sqrt(all_self_dot);
	const T l_e = sqrt(imag_self_dot);

	if (l_d != 0)
	{
		out.vd[0] = log(l_d);
	}

	if (l_e != 0)
	{
		const T x = 1 / l_e;
		const T y = acos(in.vd[0] / l_d);
		const T z = x * y;

		for (size_t i = 1; i < N; i++)
			out.vd[i] = in.vd[i] * z;
	}

	return out;
}

// New multiplication function for variable T and N
template<class T, size_t N>
vertex<T, N> mul(const vertex<T, N>& in_a, const vertex<T, N>& in_b)
{
	return exp(log(in_a) + log(in_b));
}

template<class T, size_t N>
vertex<T, N> get_traditional_commutator(vertex<T, N> in_a, vertex<T, N> in_b)
{
	vertex<T, N> AB = traditional_mul(in_a, in_b);
	vertex<T, N> BA = traditional_mul(in_b, in_a);

	vertex<T, N> C;

	for (size_t i = 0; i < N; i++)
		C.vd[i] = AB.vd[i] - BA.vd[i];

	return C;
}

template<class T, size_t N>
vertex<T, N> get_new_commutator(vertex<T, N> in_a, vertex<T, N> in_b)
{
	vertex<T, N> AB = mul(in_a, in_b);
	vertex<T, N> BA = mul(in_b, in_a);

	vertex<T, N> C;

	for (size_t i = 0; i < N; i++)
		C.vd[i] = AB.vd[i] - BA.vd[i];

	return C;
}

template<class T, size_t N>
vertex<T, N> div(const vertex<T, N>& in_a, const vertex<T, N>& in_b)
{
	// c = a/b

	// c = inv(b) * a
	// inv(b) = conjugate(b) / norm(b)
	// c = (conjugate(b) / norm(b)) * a

	T b_norm = 0;

	for (size_t i = 0; i < N; i++)
		b_norm += (in_b.vd[i] * in_b.vd[i]);

	vertex<T, N> temp_b;

	temp_b.vd[0] = in_b.vd[0] / b_norm;

	for (size_t i = 1; i < N; i++)
		temp_b.vd[i] = -in_b.vd[i] / b_norm;

	vertex<T, N> a = temp_b;
	vertex<T, N> b = in_a;

	vertex<T, N> ln_a = log(a);
	vertex<T, N> ln_b = log(b);

	vertex<T, N> out = ln_a + ln_b;
	vertex<T, N> o = exp(out);
	return o;
}

template<class T, size_t N>
vertex<T, N> sin(const vertex<T, N>& in)
{
	T all_self_dot = 0;
	T imag_self_dot = 0;
	vertex<T, N> out;

	for (size_t i = 1; i < N; i++)
		imag_self_dot += (in.vd[i] * in.vd[i]);

	all_self_dot = imag_self_dot + (in.vd[0] * in.vd[0]);

	if (all_self_dot == 0)
	{
		for (size_t i = 0; i < N; i++)
			out.vd[i] = 0;

		return out;
	}

	const T l_d = sqrt(all_self_dot);
	const T l_e = sqrt(imag_self_dot);

	out.vd[0] = std::sin(in.vd[0]) * cosh(l_e);

	if (l_e != 0)
	{
		const T x = 1 / l_e;
		const T y = cos(in.vd[0]) * sinh(l_e);
		const T z = x * y;

		for (size_t i = 1; i < N; i++)
			out.vd[i] = in.vd[i] * z;
	}

	return out;
}
