Translating the do notation into a series of >> and >>=

> do x
>    y

=

> x >> y

> do a <- x
>    y

=

> x >>= \a -> y

Translating the usage of the following divide function

> divide :: Monad m => Int -> Int -> ErrTM m Int
> divide _ 0 = eFail
> divide x y = return (x `div` y)

> divisions :: Monad m => ErrTM m [Int]
> divisions = do
>   a <- divide 10 20
>   b <- divide 30 40
>   c <- divide 10 02
>   return [a,b,c]

the divisions function becomes

> divisions = divide 10 20 >>= \a -> divide 30 40 >>= \b -> divide 10 02 >>= \c -> return [a,b,c]

Noting the definition of (>>=) for ErrTM is

> instance Monad m => Monad (ErrTM m) where
>   (ErrTM m) >>= f = ErrTM $ m >>= r
>      where unwrapErrTM (ErrTM v) = v
>            r (Just x) = unwrapErrTM $ f x
>            r Nothing = return Nothing 

So knowing the definition of (>>=) for ErrTM we can now see that the divisions function operates entirely within the ErrTM monad. The expressions on either side of the bind are evaluated, and the output of the bind itself is wrapped into the ErrTM monad. This results in the final output of the chain of binds (return [a,b,c]) being wrapped, resulting in a return type of ErrTM m [Int] and not [ErrTM m Int].

When we look at this using (>>) instead, the semantics hold, as we can define (>>) as follows:

> m >> f = m >>= \_ -> f