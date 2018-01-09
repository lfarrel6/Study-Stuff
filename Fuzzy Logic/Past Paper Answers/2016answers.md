Answers to [this paper.](https://www.tcd.ie/academicregistry/exams/assets/local/past-papers2016/CS/CS4001-1.PDF)

## Q1

a) The intersection of a crisp set and its complement is a null set, in keeping with the law of contradiction. The intersection of a 
fuzzy set, A, and its complement, !A, however is not a null set. The membership function for the intersection set is \mu_{I}(x) = min(\mu_{A}(x), 1-\mu_{!A}(x))

b) TODO

c) T-Norm
min(x,y) = x \cap y
bounded difference(x,y) = 0 \cup x+y-1
algebraic product(x,y) = x\*y

S-Norm
max(x,y) = x \cup y
bounded sum(x,y) = 1 \cap x+y
algebraic sum(x,y) = x+y - x\*y

d) The implications of these three functions is that defuzzification cannot be uniquely specified.
We can prove that the following inequalities will hold for two values a and b, in the range [0,1]:
`T_{BP} \leq T_{AP} \leq T_{min}`
`S_{max} \leq S_{AP} \leq S_{BD}`

## Q2

a) Fuzzification is the process of mapping crisp, observed values into membership values of fuzzy sets for linguistic terms.
Inference is the process of applying the rule base of the system to find the firing rules, and the degree to which they are firing.
Composition is the averaging process used to compute the effectiveness of each rule which has been triggered.
Defuzzification is the process of taking the computed fuzzy values and translating them back into a crisp value.

b) excellent salary: 1
large debt = 1

R1: 1 OR 0 -> 1
R2: 0 AND 1 -> 0
R3: 0

R1 fires with a effectiveness of 1.

So we compute Jim's risk using the membership fuinction \mu_{lowrisk}
low risk membership: 1/0, 1/10, 1/20, 0.5/30, 0/40, 0/50, 0/60, 0/70, 0/80, 0/90, 0/100

Using mean of maxima with alpha set to 1: (1\*0 + 1\*10 + 1\*20)/(1+1+1) = 10%

10% risk associated with Jim

c) using centre of area method: we add up all values times their membership, and divide by the sum of all membershipd
(1\*0)+(1\*10)+(1\*20)+(0.5\*30)+(0\*40)+(0\*50)+(0\*60)+(0\*70)+(0\*80)+(0\*90)+(0\*100)/(1+1+1+0.5+(0\*7))
=> 12.857

## Q4

a) term-set abbreviated: {pl, al, as, ps, bs, ws, nsl, sll}
apply membership function provided.

b) cardinality: sum up all membership values
support: crisp set of all non-zero values
core: crisp set of all full membership values i.e. 1
