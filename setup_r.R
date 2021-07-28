# enrivornment setup 
.libPaths( c( .libPaths(), "Rlib") )

if (!dir.exists("Rlib")){
  dir.create("Rlib")
  install.packages("tidyverse", repos="https://stat.ethz.ch/CRAN/", lib="Rlib")
  } 

