module TransEither () where

import Control.Monad

newtype TransEither l m r = TransEither { runTransEither :: m (Either l r) }

class MonadTrans t where
  lift :: Monad m => m a -> t m a

class (Show e, Monad m) => (MonadError e m) where
  eFail :: e -> m a
  eHandle :: m a -> (e -> m a) -> m a

instance (Monad m) => Functor (TransEither l m) where
  fmap = liftM

instance (Monad m) => Applicative (TransEither l m) where
  pure = TransEither . pure . Right
  (<*>) = ap

instance (Monad m) => Monad (TransEither l m) where
  return = pure
  x >>= y = TransEither $ do
    x' <- runTransEither x
    case x' of
      Left err     -> return $ Left err
      Right result -> runTransEither $ y result

instance MonadTrans (TransEither l) where
  lift m = TransEither $ do
    m' <- runTransEither
    return $ Right m'

instance (Show l, Monad m) => MonadError l (TransEither l m) where
  eFail l = TransEither $ return $ Left l
  eHandle f1 f2 = TransEither $ do
    f1' <- runTransEither f1
    case f1' of
      Right result -> return $ Right result
      Left  err    -> runTransEither $ f2 err