a) Explain what a monad transformer is, and indicate why it is necessary to have a special technique to overcome the issue with combining arbitrary monads.

-----------------------------------------------------------------------------------------------

A monad transformer is a technique used to compose monads. It achieves this by creating a monadic instance which maintains the integrity of the wrapped monad, and provides its own behaviour alongside it.

It is necessary to have a special technique to overcome the issue of combining arbitrary monads as monads don't naturally compose. Monads are a powerful tool in haskell and provide important behavioural guarantees, however their composition is quite ambigious. For example, when composing the State and Maybe monads, it is not clear whether it ought to be            State (s -> Maybe (a,s)) or State (s -> (Maybe a, s)). In many cases, there is no 'right' answer.

Monad transformers provide a robust and consistent means of monad composition.

-----------------------------------------------------------------------------------------------

b) Give an implementation for the Writer Transformer Monad (that is, give the code for a monad transformer than can add a tell operation to any monad; the operation should accumulate a list of values. It could be used, for example, to build up a string containing logging data).

-----------------------------------------------------------------------------------------------

> newtype WriterT w m a = = WriterT { runWriterT :: m (a,w) }
>
> instance (Monad m, Monoid w) => Functor (WriterT w m) where
>  fmap = liftM
>
> instance (Monad m, Monoid w) => Applicative (WriterT w m) where
>  pure a = WriterT $ return (a,mempty) 
>
> instance (Monad m, Monoid w) => Monad (WriterT w m) where 
>  return = pure
>  x >>= y = do
>              (a,w)   <- runWriterT x
>              (a',w') <- runWriterT $ y a
>              return (a', w `mappend` w')

-----------------------------------------------------------------------------------------------

c) Explain why the Haskell IO monad cannot be a monad transformer.

-----------------------------------------------------------------------------------------------

The IO monad cannot exist as a monad transformer as that would allow for several instances of the IO monad within a single stack. This would violate referential transparency. Also, having multiple instances of the IO monad in the same stack could interfere with the state of the environment held by the IO monad i.e. an action in one IO monad, may be ignored by another.

-----------------------------------------------------------------------------------------------

d) What does the lift function do in the monad transformer class?

-----------------------------------------------------------------------------------------------

The lift function brings a given monadic action/function into the correct layer of the stack. For example, in the example of Writer w State (a,s) () , to perform a get or put action in the state monad, we would need to lift the action into the State monad. In the case of the mtl library, this can be avoided assuming there is only a single instance of a given monad in the stack. The transformers library requires explicit lifting.

-----------------------------------------------------------------------------------------------

e) Sketch how the Haskell mtl libraries use type classes to avoid requiring programmers to make explicit reference to the lift operation.

-----------------------------------------------------------------------------------------------

The mtl library creates a type class for each transformer. By doing this, any monad transformer which interacts with mtl library transformers must be an instance of each transformers type class.
For example, if I made a monad transformer called OptimusPrimeT and needed to use it in a stack with an instance of the State monad, then OptimusPrimeT would need to be an instance of MonadState in order to lift the state actions into the state monad (without explicit lifting).