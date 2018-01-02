module TransMaybe () where

import Control.Monad

newtype TransMaybe m a = TransMaybe { runTransMaybe :: m (Maybe a) }

class MonadTrans t where
  lift :: m a -> t m a

class Monad m => MonadError m where
  eFail   :: m a
  eHandle :: m a -> m a -> m a

instance Monad m => Functor (TransMaybe m) where
  fmap = liftM

instance Monad m => Applicative (TransMaybe m) where
  pure = TransMaybe . return . Just
  (<*>) = ap

instance Monad m => Monad (TransMaybe m) where
  return = pure
  m >>= k = TransMaybe $ do
    r <- runTransMaybe m
    case r of
     Nothing -> return Nothing
     Just x  -> runTransMaybe $ k x

instance MonadTrans TransMaybe where
  lift m = TransMaybe $ do
    m' <- m
    return $ Just m'

instance Monad m => MonadError (TransMaybe m) where
  eFail   = TransMaybe $ return Nothing
  eHandle f1 f2 = TransMaybe $ do
    f1' <- runTransMaybe f1
    case f1' of
      Just _  -> return f1'
      Nothing -> runTransMaybe $ f2