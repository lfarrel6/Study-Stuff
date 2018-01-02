module TransWriter () where

import Control.Monad
import Control.Monad.Writer

newtype TransWriter w m a = TransWriter { runTransWriter :: m (a,w) }

instance (Monad m,Monoid w) => Functor (TransWriter w m) where
  fmap = liftM

instance (Monad m, Monoid w) => Applicative (TransWriter w m) where
  pure a = TransWriter $ pure (a,mempty)

instance (Monad m, Monoid w) => Monad (TransWriter w m) where
  return = pure
  m >>= k = TransWriter $ do
   (a,w) <- runTransWriter m
   (a',w') <- runTransWriter $ k a
   return (a',w `mappend` w')