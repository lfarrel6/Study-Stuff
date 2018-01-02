{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module TransState () where

import Control.Monad
import Control.Monad.State

newtype TransState s m a = TransState { runTransState :: s -> m (a,s) }

instance Monad m => Functor (TransState s m) where
  fmap = liftM

instance Monad m => Applicative (TransState s m) where
  pure a = TransState $ \x -> return (a,x)

instance Monad m => Monad (TransState s m) where
  return = pure
  f1 >>= f2 = TransState $ \s -> do
   (a,s') <- runTransState f1 s
   runTransState (f2 a) s'