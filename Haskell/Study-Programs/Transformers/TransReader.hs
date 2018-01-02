module TransReader () where

import Control.Monad

newtype TransReader r m a = TransReader { runTransReader :: r -> m a }

class MonadTrans t where
  lift :: m a -> t m a

class Monad m => MonadReader r m where
  ask  :: m r
  asks :: (r -> a) -> m a

instance Monad m => Functor (TransReader r m) where
  fmap = liftM

instance Monad m => Applicative (TransReader r m) where
  pure a = TransReader $ \_ -> return a
  (<*>)  = ap

instance Monad m => Monad (TransReader r m) where
  return = pure
  x >>= y = TransReader $ \ r -> do
    x' <- runTransReader x r
    runTransReader (y x') r

instance MonadTrans (TransReader r) where
  lift m = TransReader $ \_ -> m

instance Monad m => MonadReader r (TransReader r m) where
  ask    = TransReader $ \r -> return r
  asks a = TransReader $ \r -> return $ a r