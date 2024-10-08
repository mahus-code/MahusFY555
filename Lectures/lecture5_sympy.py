import sympy as sym

x = sym.symbols('x')
f = sym.symbols('f', cls = sym.Function)

diffeq = sym.Eq(f(x).diff(x), -0.5*f(x) )

sol = sym.dsolve(diffeq, f(x))
print(sol)