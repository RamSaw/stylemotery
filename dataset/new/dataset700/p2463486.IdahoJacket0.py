import sys
import math

def is_palindrome( number ):
	num_str = str( number )
	num_len = len( num_str )
	half_len = int( num_len + 1 / 2 )
	end = num_str[-half_len:]
	rev_end = end[::-1]
	start = num_str[:half_len]
	equal = rev_end == start      
	return equal

for x in range( 1, int( math.sqrt( 10e14 ) ) ):
	if ( is_palindrome( x ) and is_palindrome( x * x ) ):
		print(x * x)
