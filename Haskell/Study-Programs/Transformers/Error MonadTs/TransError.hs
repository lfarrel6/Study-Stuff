module TransError () where

import Control.Monad

newtype TransError e m a = TransError { runTransError :: m (Either e a)}

class MonadTrans t where
  lift :: m a -> t m a

class Monad m => MonadException e m where
  throw :: e -> m a
  catch :: m a -> (e -> m a) -> m a

instance Monad m => Functor (TransError e m) where
  fmap = liftM

instance Monad m => Applicative (TransError e m) where
  pure = TransError . return . Right
  (<*>) = ap

instance Monad m => Monad (TransError e m) where
  return = pure
  x >>= y = TransError $ do
    x' <- runTransError x
    case x' of
      Right res -> runTransError $ y res
      Left  _   -> return x'

instance MonadTrans (TransError e) where
  lift m = TransError $ do
    m' <- runTransError
    return $ Right m'

instance Monad m => MonadException e (TransError e m) where
  throw = TransError . return . Left
  catch f1 f2 = TransError $ do
    f1' <- runTransError f1
    case f1' of
      Left err -> runTransError $ f2 err
      Right _  -> return f1'