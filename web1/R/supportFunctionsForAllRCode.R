# this is R code/functions I commonly use for all R code
#========================================================================================
# Get error codes from universal file
#========================================================================================#

# Nor completely sure how this does it, but it gets the full name of the file calling it
thisFile <- function() {
        cmdArgs <- commandArgs(trailingOnly = FALSE)
        needle <- "--file="
        match <- grep(needle, cmdArgs)
        if (length(match) > 0) {
                # Rscript
                return(normalizePath(sub(needle, "", cmdArgs[match])))
        } else {
                # 'source'd via R console
                return(normalizePath(sys.frames()[[1]]$ofile))
        }
}

# this just takes care of some file name handling to call Carl's path_helper.py
callPathHelper <- function(searchTerm){
    # first we get the directory this file is saved in
    dirOfCallingScript <- dirname(thisFile())
    dirsInPath <- unlist(strsplit(dirOfCallingScript ,.Platform$file.sep))
    web1Ind <- grep('web1', dirsInPath)
    web1Dir <- paste(dirsInPath[1:web1Ind], collapse = .Platform$file.sep)
    # and we know that pathHelper is one above this
    filePath <- system(paste0(web1Dir, "/path_helper.py ", searchTerm), intern = TRUE)
    return(filePath)
}

# find the error code file, then read it in, and create a single row df
getErrorCodes <- function(){
    exitCodeFile <- callPathHelper('exit_codes')
    exitCodesDF <- read.table(exitCodeFile,
                              header = FALSE,
                              sep = "\t",
                              stringsAsFactors = FALSE,
                              fill = TRUE
                             )
    exitCodes <- as.vector(as.numeric(exitCodesDF[,1]))
    names(exitCodes) <- exitCodesDF[,2]
    return(exitCodes)
}

# one last step to make this accessible in all subsequent functions/files/commands
exitCodes <- getErrorCodes()


#========================================================================================
# General support functions
#========================================================================================#

# once either sig or meta GEO have run successfully all the way through, we want to be able to see the library versions used
finishedSuccessfully <- function() {
    print("Session info: ")
    print(sessionInfo())
    print("Finished successfully")
}

# It's nice to be able to get the number of lines in an outside file
getLineNumberFromFile <- function(fileToCheck){
    return(as.numeric(system(paste0("cat \"", fileToCheck, "\" | wc -l | tr -d ' '"), intern = TRUE)))
}

# Another nice little function, this one to strip leading and trailing white spaces
trimWhiteSpaces <- function (x) gsub("^\\s+|\\s+$", "", x)

# short hand to go from factor to number
asNum <- function(stringNumber){
    return(as.numeric(as.character(stringNumber)))
}
