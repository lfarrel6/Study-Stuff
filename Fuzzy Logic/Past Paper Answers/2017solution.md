## My answers to [this paper.](https://www.tcd.ie/academicregistry/exams/assets/local/past-papers2017/CS/CS4001-1.PDF)

## Q1

a) No a crisp set cannot be used to described Bertie's mammals completely - crisp sets only work if membership is binary.
Bertie's description of his mammals includes the discussion of half-mammals, and 10% mammals.
Crisp set theory does not capture this concept of elastic membership.

b) A^{mammals} = 1/horses + 1/zebras + 1/tigers + 0.5/platypus + 0.5/echidnas + 0.1/whales + 0.1/dolphins + 0/sparrows + 0/robins + 0/crows + 0/penguins + 0/kiwis
A'^{mammals} = 0/horses + 0/zebras + 0/tigers + 0.5/platypus + 0.5/echidnas + 0.9/whales + 0.9/dolphins + 1/sparrows + 1/robins + 1/crows + 1/penguins + 1/kiwis
E^{intersect} = 0/horses + 0/zebras + 0/tigers + 0.5/platypus + 0.5/echidnas + 0.1/whales + 0.1/dolphins + 0/sparrows + 0/robins + 0/crows + 0/penguins + 0/kiwis
No, E^{intersect} does not obey the classic laws of logical thought as it violates the law of contradiction.
The law of contradiction states that the intersection between a set and its complement must be a null set. However we can see that E^{intersect} is not a null set.

c) The three most common operations used for logical conjunction in fuzzy systems are:
`T_{min}(x,y) = x \cap y`
`T_{algebraic product}(x,y) = x \times y`
`T_{bounded difference}(x,y) = 0 \cup x+y-1`


## Q2

a) The notion of being tall: can be represented using a sigmoid function, as the membership to the set increases until it reaches certainty
at which point it continues as a certainty (1): \mu_{tall}(x) = \frac{1}{1+\e^{-a*(x-c)}}

About-ness: aboutness can be represented using a symmetric triangular function
``` \mu_{about}(x) = 1 - \vbar \frac{m-x}{d} \vbar if x is \leq m+d and x \geq m-d 
                     else if x > m+d or x < m-d then 0 ```

Approximately between: the notion of a value being approximately between two values can be represented using a trapezoidal membership funciton
\mu_{between}(x) = \max{\min{ \frac{x-a}{b-a},1,\frac{d-x}{d-c} },0}

b) The appropriateness of the velocity of a rotation disk can be captured into a fuzzy set according to the trapezoidal membership function:
``` \mu_{velocity}(x) = \max{\min{ \frac{x-a}{b-a},1,\frac{d-x}{d-c} },0} where a is the speed below which is should not fall,
b is the lower acceptable velocity, c is the upper acceptable velocity, and d is the maximum speed which it should not exceed.
```
The core values of the velocity fuzzy set is the subset of acceptable values for which the membership to the velocity set is 1.
The support values are all values in the velocity set which have non-zero membership values i.e. all values which are within the range of the upper and lower limits.

No, a machine with a velocity function with no core would have no concept of an optimal speed and so would fail to work.

c)
Classic | Fuzzy
--------|-------
if A then B | if A(\mu_{A}) then B(\mu_{B})
A is-a-part-of B | for all x of X, \mu_{A}(x) \leq \mu_{B}(x)
A:=5 and if A < 5 then B:=A+5 | ?
A weighs 5 kg | A weighs about 5kg
A is false | \mu_{false-A}(x) = 1- \mu_{A}(x)
A belongs-to-class B, so !A does-not-belong-to B | (this is an expression of the law of contradiction, which isn't obeyed in fuzzy logic) \mu_{B}(A) > 0 , \mu_{B}(!A) = 1 - \mu_{B}(A)
