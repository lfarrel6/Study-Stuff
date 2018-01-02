{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module TransState () where

import Control.Monad
import Control.Monad.State

newtype TransState s m a = TransState { runTransState :: s -> m (a,s) }

class MonadTrans t where
  lift :: m a -> t m a

class (MonadState s m) where
  get    :: m s
  put    :: s -> m ()
  modify :: (s -> s) -> m ()

instance Monad m => Functor (TransState s m) where
  fmap = liftM

instance Monad m => Applicative (TransState s m) where
  pure a = TransState $ \x -> return (a,x)
  (<*>) = ap

instance Monad m => Monad (TransState s m) where
  return = pure
  f1 >>= f2 = TransState $ \s -> do
   (a,s') <- runTransState f1 s
   runTransState (f2 a) s'

instance MonadTrans (TransState s) where
  lift m = TransState $ \s -> do
    m' <- m
    return (m',s)

instance Monad m => MonadState s (TransState s m) where
  get = TransState $ \s -> return (s,s)
  put s = TransState $ \_ -> return ((),s)
  modify f = get >>= put . f