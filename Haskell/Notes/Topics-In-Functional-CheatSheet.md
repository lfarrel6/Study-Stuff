### An attempt to condense the topics on the course into one file, if anything isn't clear, [read the full notes](https://github.com/lfarrel6/Study-Stuff/tree/master/Haskell/Notes)

### (Examinable) Topics

  - [Monads](#monads)
  - [Monad Transformers](#monad-transformers)
  - [Parallelism](#parallelism)
  - [Concurrency](#concurrency)
  - [Embedded DSLs](#embedded-dsls)
  - [Implementing Type Inference](#implementing-type-inference)
  - [Advanced Type Systems](#advanced-type-systems)
  
### Monads

##### What are monads?
  - [Composable computation descriptions](https://wiki.haskell.org/Monad)
  - Used to structure programs, maintain referential transparency, and group actions with common behaviours.
  
  - Minimum Requirements for a monadic instance
    * `return` - a simple function which takes a value and lifts it into a monad (`return :: a -> m a`)
    * `>>=` - a.k.a bind - used to compose monadic computations (`(>>=) :: m a -> (a -> m b) -> m b`)
    
  - Examples of Monads:
    * `IO` - used to perform what would otherwise be unsafe IO actions
    * `State` - used to provide the concept of a state, and the result of some 'stateful' computations
    * `Maybe` - used to handle the concept of failure in computations, successful computations result in a value wrapped by `Just`, and failure results in `Nothing`
    * `Either` - similar to Maybe but returns a value on failure, successful computation returns a `Right` wrapped value, while failure returns a `Left` wrapped value (typically a string explaining the error)
    * `[]` - lists in Haskell are also monads. Compare to maybe: can fail and give `[]` (which is nothing), or succeed and give some values (like `Just`)

  - Monad Laws
    * Left Identity: `return a >>= f = f a`
    * Right Identity: `m >>= return = m`
    * Associativity: `(m >>= f) >>= g = m >>= (\x -> f x >>= g)`
    * **Simply put:** [monad composition is an associative operator with left and right identities.](https://wiki.haskell.org/Monad_laws)
    
  - Applicative Functors
    * GHC 7.10 added new abstraction: Monad class was refactored to **Functor**, **Applicative**, and **Monad**
  
  - Functors (each mention of f refers to a Functor)
    * Class for types which can be mapped over, allows us to apply functions to wrapped (functorial) values
    * Provides `fmap :: (a -> b) -> (f a -> f b)` which can be used as an infix operator using `<$>`
    * `fmap` takes a function `(a -> b)` and lifts it into the context of a wrapped value `(f a -> f b)`
    * **First Functor Law:**  `fmap id = id`
    * **Second Functor Law:** `fmap (g. h) = fmap g . fmap h`
    
  - Applicatives (each mention of f refers to an Applicative, the term morphism refers to `f (a -> b)`)
    * More Structured than functors
    * Provides `<*>` which essentially lets us apply more args to `fmap` arbitrarily
    * `(<*>) :: f (a -> b) -> f a -> f b`
    * Also provides `pure` which wraps its argument into the correct type
    * `pure :: a -> f a`
    * **First Applicative Law - Identity:** Applicatives preserve identity. `pure id <*> v = v`
    * **Second Applicative Law - Homomorphism:** Applicatives preserve function application. `pure f <*> pure x = pure (f x)`
    * **Third Applicative Law - Interchange:** Applying a morphism, u, to a `pure` value y is the same as applying `pure ($ y)` to the morphism. `u <*> pure y = pure ($ y) <*> u`
    * **Fourth Applicative Law - Composition:** Applicatives maintain composition (i.e. `pure (.)` acts on morphism as `(.)` acts on functions)`pure (.) <*> u <*> v <*> w = u <*> (v <*> w)`

  - Monads (Post GHC 7.10)
    * **All Monads are also Functors and Applicatives**
    * `(>>=) :: m a -> (a -> m b) -> m b`
    * `return :: a -> m a`
    * `fail :: String -> m a`
    * Monads post 7.10 provide the same functionality, Functors and Applicatives just capture some of the more general behaviour
    * Functors and Applicatives can make defining a monadic instance easier: `return = pure` is a satisfactory definition
    * And if defining the instance of Functor for a Monad, we can also use `fmap = liftM`

##### [Further explanation of Functors, Applicatives, and Monads, and their relationship.](https://en.wikibooks.org/wiki/Haskell/Applicative_functors)

##### A nice extract from this article :point_up:
  - **Functor, Applicative, Monad - A sliding scale of power**
  - Ignoring `pure`/`return`, each class can be said to have a characteristic method
    * `fmap :: Functor f => (a -> b) -> f a -> f b`
    * `(<*>) :: Applicative f => f (a -> b) -> f a -> f b`
    * `(>>=) :: Monad m => m a -> (a -> m b) -> m b`
  - They all look very different... **but lets change fmap to its infix synonym `(<$>)`, and flip `(>>=)` to get `(=<<)`**
    * `(<$>) :: Functor t     =>   (a ->   b) -> (f a -> f b)`
    * `(<*>) :: Applicative t => t (a ->   b) -> (f a -> f b)`
    * `(=<<) :: Monad t       =>   (a -> t b) -> (t a -> t b)`
  - All the type signatures line up and are very similar! So now we can see the similarities, as well as the differences
    * `fmap` maps arbitrary functions over functors
    * `(<*>)` maps `t (a -> b)` *morphisms* over applicative functors
    * and `(=<<)` maps `a -> t b` functions over monadic functors
  - The influence of the types of `fmap`, `(<*>)`, `(>>=)`
    * `fmap` ensures that we cannot change the context no matter what function we give it. This is because the `(a -> b)` argument has nothing to do with the functorial context t, so its application can have no influence on that context.
    * So `fmap` gives the guarantee of safety. For example, `fmap f xs` where `xs` is some list can **never** change the number of elements of the list.
    * `(<*>)` then gives us a way to apply functions with context i.e. morphisms. This lets us change context. For example, `[(2*),(3*)] <*> [2,5,6]` creates a list of 6 elements (which isn't possible with a Functor).
    * We combine a morphism with its own context, `t (a -> b)` and combine it with a functorial value `t a`. This has a subtlety to it. While `t (a -> b)` is within context `t`, the `(a -> b)` of this morphism cannot modify the context. **So applicatives can perform fully deterministic context changes, restricted by the context of the argument.**
    * A simple way to think about it, using the `[(2*),(3*)] <*> [2,5,6]` example: knowing that the use of `<*>` on list results in the application of every function to every value, we can *definitively* say what the length of the resulting list will be based on the lengths of the inputs. I.e. `(<*>)` applied to a list of length 2, and a list of length 3, is *guaranteed* to give a list of length 6. We lose this guarantee when we move into monads.
  * `(>>=)` takes in a `(a -> t b)` **which can create context from values!** This provides flexibility.
  * Monads introduce a knock-on effect of computation: the result of the first computation may influence the second. Think about binding `Maybe` computations, if the first evaluates to `Nothing` then we won't even attempt the second computation.

##### How would we let the compiler know that the Functor we are defining is also a Monad?

##### Haskell Type Classes

  - Haskell uses type classes to explicitly restrict polymorphic types to have certain behaviours
  - Example: Equality
    * The type of equality (intuitively) is `(==) :: a -> a -> Bool` - i.e. for all types a, we take two a values and return a boolean of their equality
    * In haskell, the type is `(==) :: (Eq a) => a -> a -> Bool` - i.e. for all types a which are part of the Equality class, take two values and return a boolean of their equality
  - This is **Ad-Hoc polymorphism**
    * Equality is polymorphic (using `a`)
    * But it is ad-hoc - it requires a specific implementation of the `Eq` methods for a given type
    * In contrast, list length displays parametric polymorphism (no type dependency) `length [] = 0 | (x:xs) = 1 + length xs`
    * Ad-hoc polymorphism is everywhere: the overloading of `+`
    * Overloading is built into many languages, but is in Haskell as a language feature (type classes) - can easily create new instances
    * If we wanted to define our own: Specify the name/operator (==) , Describe its pattern of use (`a -> a -> Bool`) , Provide a class name for the concept (Eq)
    * To then use our operator with a given type (e.g. Bool) we would provide an implementation for that type
  - How does Haskell interpret class name/operator?
    * Notes the association between symbol and class
    * Deduces the type of the args, and verifies that there is an instance of the type for the class
    * If it is well typed: generates the appropriate code
    * Else: suggests you add an instance declaration for your type in the class

##### Some standard prelude classes

Type of Behaviour | Classes
----|----
Relation | Eq, Ord
Enumeration | Enum, Bounded
Numeric | Num, Real, Integral, Fractional, Floating, RealFrac, RealFloat
Textual | Show, Read
**Categorical** | **Functor, Monad**

### Monad Transformers

### Parallelism

### Concurrency

### Embedded DSLs

### Implementing Type Inference

### Advanced Type Systems
