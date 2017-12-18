module Arithmetic ( evaluate , interpret , Equation(..) ) where

import Text.Read          (read)

data Equation = Mul Double
              | Add Double
              | Sub Double
              | Div Double
              | Exit

evaluate :: Double -> Equation -> Double
evaluate x (Mul y) = x*y
evaluate x (Add y) = x+y
evaluate x (Sub y) = x-y
evaluate x (Div y) = x/y

interpret :: [String] -> Equation
interpret ["*",y] = Mul (read y :: Double)
interpret ["+",y] = Add (read y :: Double)
interpret ["-",y] = Sub (read y :: Double)
interpret ["/",y] = Div (read y :: Double)
interpret _       = Exit
{-
eval :: Num a => [Equation] -> a
eval (x:xs) = evaluate x

evalSeq :: [String] -> [Equation]
evalSeq [a,b,c]    = (interpret a b c)
evalSeq (a:b:c:es) = (interpret a b c):(evalSeq es)
evalSeq _          = []
-}