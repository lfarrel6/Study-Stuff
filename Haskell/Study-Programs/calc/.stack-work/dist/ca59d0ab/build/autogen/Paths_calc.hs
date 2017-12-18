{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -fno-warn-implicit-prelude #-}
module Paths_calc (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "C:\\Users\\Rory\\Desktop\\Study-Stuff-Repo\\Haskell\\Study-Programs\\calc\\.stack-work\\install\\69301fa4\\bin"
libdir     = "C:\\Users\\Rory\\Desktop\\Study-Stuff-Repo\\Haskell\\Study-Programs\\calc\\.stack-work\\install\\69301fa4\\lib\\x86_64-windows-ghc-8.0.2\\calc-0.1.0.0-9Oq6hdcIFCX6Vb7F86N9Ql"
dynlibdir  = "C:\\Users\\Rory\\Desktop\\Study-Stuff-Repo\\Haskell\\Study-Programs\\calc\\.stack-work\\install\\69301fa4\\lib\\x86_64-windows-ghc-8.0.2"
datadir    = "C:\\Users\\Rory\\Desktop\\Study-Stuff-Repo\\Haskell\\Study-Programs\\calc\\.stack-work\\install\\69301fa4\\share\\x86_64-windows-ghc-8.0.2\\calc-0.1.0.0"
libexecdir = "C:\\Users\\Rory\\Desktop\\Study-Stuff-Repo\\Haskell\\Study-Programs\\calc\\.stack-work\\install\\69301fa4\\libexec"
sysconfdir = "C:\\Users\\Rory\\Desktop\\Study-Stuff-Repo\\Haskell\\Study-Programs\\calc\\.stack-work\\install\\69301fa4\\etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "calc_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "calc_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "calc_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "calc_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "calc_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "calc_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "\\" ++ name)
