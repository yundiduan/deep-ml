from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""

	n = len(point)
	ans = []
	for i in range(n):
		row = []
		for j in range(n):
			d = 0
			point[i] += h
			point[j] += h
			d += f(point)
			point[j] -= 2 * h
			d -= f(point)
			point[i] -= 2 * h
			d += f(point)
			point[j] += 2 * h
			d -= f(point)
			point[i] += h
			point[j] -= h
			row.append(d / 4 / h / h)
		ans.append(row)
	return ans


