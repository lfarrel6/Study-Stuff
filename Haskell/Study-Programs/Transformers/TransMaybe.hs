module TransMaybe () where

import Control.Monad

newtype TransMaybe m a = TransMaybe { runTransMaybe :: m (Maybe a) }

instance Monad m => Functor (TransMaybe m) where
  fmap = liftM

instance Monad m => Applicative (TransMaybe m) where
  pure = return

instance Monad m => Monad (TransMaybe m) where
  return = TransMaybe . return . Just 
  m >>= k = TransMaybe $ do
    r <- runTransMaybe m
    case r of
     Nothing -> return Nothing
     Just x  -> runTransMaybe $ k x