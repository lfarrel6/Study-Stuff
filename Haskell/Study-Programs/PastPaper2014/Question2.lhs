a) Explain how Generalised Algebraic Data Types (GADTs) in Haskell allow programmers to express constraints on data types, and suggest why this may be a useful feature for programmers.

-----------------------------------------------------------------------------------------------

GADTs allow users to give explicit type signatures for data type constructor functions. This is an extremely powerful tool, which gives the programmer more control over the types of their data types, and also removes a level of abstraction by showing that a data type is a constructor function.

Consider the example of a expression evaluation system.
It can represent ints and bools, and can perform addition and conditional evaluation (if statements).

The naive implementation would suggest something like this:

> data Expr = I Int
>           | B Bool
>           | Add Expr Expr
>           | If Expr Expr Expr

However, this accepts the Expr 'Add (B True) (I 5)' even though this clearly violates the semantics of the expression system, and is not (obviously) computable.

Using GADTs this can be avoided:

> data Expr a where
>   I   :: Int  -> Expr Int
>   B   :: Bool -> Expr Bool
>   Add :: Expr Int -> Expr Int -> Expr Int
>   If  :: Expr Bool -> Expr a -> Expr a -> Expr a

Using the above definition of Expr, 'Add (B True) (I 5)' would be rejected. The programmer has the ability to restrict the types of the constructors further.

-----------------------------------------------------------------------------------------------

b) Give a definition of a list type in Haskell that can be used to store values of differing types (that is, a list where it is not necessary to have each element of the same type).

Explain what problem a programmer will encounter when attempting to operate on elements of such a list and describe two mechanisms that could be used to overcome this problem. For each mechanism indicate what the trade-offs the programmer will enounter might be.

-----------------------------------------------------------------------------------------------

This is achieved using Existential types.

> data MyList = MyNil 
>             | MyCons a MyList

By not mentioning the type a in the data type itself, each element within the list can contain some a which has no requirement to match the type of another list element. `This is known as existential quantification.`

The type checker will not be able to succesfuly quantify some a across each element in the list, so any class such as show will not be possible for the items of this list.

We could constrain the values allowed into the list using a class, however this restricts the possible entries to this class. For example, to allow for values from show:

> data MyList = MyNil
>             | Show a => MyCons a MyList

If the restriction to types which have an instance of show is too restrictive, we could package the values with functions to make them useful. This is effective however it is a hugely cumbersome approach:

> data MyList = MyNil
>             | MyCons (a, (a -> String)) MyList

-----------------------------------------------------------------------------------------------

c) Dependent type systems allow for an even greater degree of expression than GADTs. Explain why this is, and suggest why language designers may be reluctant to incorporate dependent types into their programming languages despite this.

-----------------------------------------------------------------------------------------------

Dependent type systems allow for data types to depend on an attribute of theirs, for example a list may depend on its length. Such a system can increase the level of safety in a programming language. For example, with a dependent type system, it would be possible to have a safe implementation of head for lists, something which haskell cannot do.

There are ways to simulate dependent type systems in haskell, using the data kinds and type kinds extensions. Using data kinds, the length of a list type, can be promoted into type level and using type kinds, operations can be defined onto these type level values. This facilitates the implementation of safe head, however defining the list to be dependent on its length makes it far harder to reason about, and use the list.

`insert safe head implementation`