Answers for [this paper.](https://www.tcd.ie/academicregistry/exams/assets/local/past-papers2015/CS/CS4001-1.PDF)

## Q1

a) The intersection of a crisp set and its complement is a null set, in keeping with the law of contradiction. The intersection of a fuzzy set, A, and its complement, !A, however is not a null set. The membership function for the intersection set is \mu_{I}(x) = min(\mu_{A}(x), 1-\mu_{!A}(x))

b) Commutativity: T(x,y) = T(y,x)
Associativity: T(x,T(y,z)) = T(T(x,y),z)
Monotonicity: if y \leq z -> T(x,y) \leq T(x,z)
Boundary: T(x,1) = x

c) T-norm
min: a \cap b
algebraic product: a\*b
bounded difference: 0 \cup a+b-1

S-norm
max: a \cup b
algebraic sum: a + b - ab
bounded sum: 1 \cap a + b

d) Defuzzification process cannot be uniquely defined. 

## Q2

a) I answered this on paper.

b) this also.

## Q3

a) i) Mamdani: If X is <sub>x</sub>(X) then Y is \mu(Y)
      TSK: If X is <sub>x</sub>(X) then Y is f(X)
  
   ii) Mamdani: <sub>x</sub>(X) & \mu(Y) are membership functions of the terms X and Y
       TSK <sub>x</sub>(X) is a membership function of term (X), and Y is a linear function of X.
       
b) In a TSK system the antecedents are reasoned about through a membership function, and the outputs are generated using a linear function
in terms of these inputs. This requires the system designers to calculate the constants required for this linear function, which can be difficult to see.
These linear functions do make computation more efficient however.

## Q4

a) done on paper, just draw out fuzzy patches to generate the rules for the system, using speed as input and voltage as output

b) to approximate the behaviour of this mamdani controller to a zero order TSK controller:
Convert our liguistic term output into a crisp value.
`IF SPEED is TOO_SLOW THEN VOLTAGE is 2.44`
`IF SPEED is RIGHT THEN VOLTAGE is 2.40`
`IF SPEED is TOO_FAST THEN VOLTAGE is 2.36`

to approximate the behaviour to a first order TSK controller:
