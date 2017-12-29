Generalised Algebraic Datatypes (GADTs) are datatypes for which a constructor has a non standard type.
They allow the programmer to specify the types of the data constructor, giving them greater power. A common use of GADTs is to increase the safety of a DSL.

Naive implementation of an expression type

> data Expr a = Num Int
>             | Boolean Bool
>             | Add Expr Expr
>             | If Expr Expr Expr

The compiler assigns standard types to each constructor:
Num     :: Int    -> Expr a
Boolean :: Bool   -> Expr a
Add     :: Expr a -> Expr a -> Expr a
If      :: Expr a -> Expr a -> Expr a -> Expr a

Using these type signatures, we can see that the compiler would accept something like Add (Num 5) (Boolean True) despite this being totally wrong.

So we need a non-standard type definition for our expressions.

> data Expr a where
>   Num     :: Int  -> Expr Int
>   Boolean :: Bool -> Expr Bool
>   Add     :: Expr Int  -> Expr Int -> Expr Int
>   If      :: Expr Bool -> Expr a   -> Expr a -> Expr a

Simples.