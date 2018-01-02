module TransWriter () where

import Control.Monad
import Control.Monad.Writer
import Data.Monoid

newtype TransWriter w m a = TransWriter { runTransWriter :: m (a,w) }

class MonadTrans t where
  lift :: m a -> t m a

class (Monoid w, Monad m) => (MonadWriter w m) where
  tell :: (Monoid w, Monad m) => w -> m ()

instance (Monad m,Monoid w) => Functor (TransWriter w m) where
  fmap = liftM

instance (Monad m, Monoid w) => Applicative (TransWriter w m) where
  pure a = TransWriter $ pure (a,mempty)
  (<*>) = ap

instance (Monad m, Monoid w) => Monad (TransWriter w m) where
  return = pure
  m >>= k = TransWriter $ do
   (a,w) <- runTransWriter m
   (a',w') <- runTransWriter $ k a
   return (a',w `mappend` w')

instance Monoid w => MonadTrans (TransWriter w) where
  lift = TransWriter . liftM (\x -> (x,mempty))

instance (Monoid w, Monad m) => MonadWriter w (TransWriter w m) where
  tell w = TransWriter $ return ((),w)