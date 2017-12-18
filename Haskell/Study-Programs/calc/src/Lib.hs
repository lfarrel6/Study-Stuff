module Lib ( someFunc ) where

import Control.Monad.IO.Class              (liftIO)
import Control.Monad.Trans.Class           (lift)
import Data.Functor.Identity               (Identity, runIdentity)
import Control.Monad.Trans.Reader          (ReaderT, ask, runReaderT)
import Control.Monad.Trans.Writer          (WriterT, tell, runWriterT)
import Control.Monad.Trans.Maybe           (MaybeT, runMaybeT)
import Control.Monad.Trans.State           (StateT, runStateT, get, put)
import Text.Read                           (readMaybe)

import Arithmetic

type Input      = Integer
type Output     = [String]
type CalcState  = Double

-- monad stack types
type UncertainIO = MaybeT IO
type MyStack = WriterT Output (StateT CalcState (UncertainIO))

-- run calculator
someFunc :: IO ()
someFunc = do
  let newState = runWriterT runCalculator
  let newMaybe = runStateT newState 1
  runMaybeT newMaybe
  putStrLn "Calculator shutting down"

-- << Simple Calculator

-- Takes in expressions in the form "op n" where op is one of the following: {*,+,-,/}
-- and n is a numeric value. The expression is computed using the state value as the 
-- second argument.
runCalculator :: MyStack ()
runCalculator = do
  liftIO $ putStrLn "=>Welcome to my monad stack calculator.\n=>It only handles one equation at a time but has a temporary memory.\n=>Enjoy."
  tell ["Greeting sent, starting calculator loop."]
  exec
 where
  exec :: MyStack ()
  exec = do
    entered <- liftIO $ getInput
    tell ["Input received: " ++ entered]
    let eq  = interpret $ words entered
    case eq of
     Exit -> tell ["Exiting."]
     _    -> do memVal <- lift $ get
                let res = evaluate memVal eq
                tell ["Input evaluated: " ++ show res]
                lift $ put res
                liftIO $ putStrLn $ show memVal ++ " " ++ entered ++ " = " ++ show res
                exec



getInput :: IO String
getInput = getLine

-- >>

{---------TESTER FUNCTIONS----------}

-- These simple functions were used to get to grips with transformers

checkNumber :: UncertainIO CalcState
checkNumber = do s <- liftIO $ getLine
                 let n = (readMaybe s)
                 case n of Just x  -> return x
                           Nothing -> return 0

-- MyStack encapsulates the Writer, State, Maybe, and IO monads.
-- Writer used for logging.
-- State used for simple memory.
-- Maybe to be used for error handling.
-- IO used as base monad and also general IO things.

-- This just takes in a value, logs it, verifies it is a number
-- and adds the value to the memory/state value
inMaybeOut :: MyStack () 
inMaybeOut = do liftIO $ putStrLn "Enter a value to add:"
                tell ["Asked for value"]
                s   <- lift $ get
                res <- lift $ lift $ checkNumber
                tell ["Given: " ++ show res]
                let newVal = s + res
                lift $ put newVal
                tell ["Stored new state: " ++ show newVal]
                liftIO $ putStrLn $ "Your result is: " ++ show newVal