T = int(input().strip())

for i in range(T):
	C, F, X = list(map(float, input().strip().split(' ')))
	best = X / 2.0
	c_sum = 0
	factories = 1
	n_sum = c_sum + C / (2.0 + (factories - 1) * F)
	while n_sum + X / (2.0 + factories * F) < best:
		best = n_sum + X / (2.0 + factories * F)
		c_sum = n_sum
		factories += 1
		n_sum = c_sum + C / (2.0 + (factories - 1) * F)

	print("Case #%s: %s" % (i + 1, best))
