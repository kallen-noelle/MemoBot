n=int(input())
def tmp(n):
	if n==0:return 0
	else :
		return n**3+tmp(n-1)
print(f"正整数{n}的前{n}项的三次方和为{tmp(n)}。")
