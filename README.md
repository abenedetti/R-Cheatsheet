# R-Cheatsheet

Often I'm googling for the very same coding issues! Think that it would be a good idea to start my own R-cheatsheet. :grin:

### 1) R Markdown special characters
coding: `<br>
tilde: ~  (ALT + 1 2 6)
<br>

### 2) Change value of a categorical variable in a dataframe
`levels(df$column)[levels(df$column)=='Old Value'] <- 'New Value'`
<br>

### 3) Create a non empty dataframe
`column1 <- c(1,2,3,4,5)`<br>
`column2 <- c('hi', 'have', 'a', 'nice', 'day')`<br>
`column3 <- c(TRUE, TRUE, FALSE, TRUE, FALSE)`<br>
`df <- data.frame(column1, column2, column3)`
<br>

### 4) Create a empty dataframe
`df <- data.frame(column1=character(), column2=integer(), column3=character())`<br>
For each column you're able to set a data type.
<br>

### 5) Check for R environment infos
`version`<br>
`sessionInfo()`
<br>

### 6) Rename dataframe column
`colnames(df)[1] <- "NewColName"`
<br>

### 7) Working with [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix)
`library(Matrix)`<br>
`sparseMat <- sparseMatrix(i = as.integer(df$Col1), j = as.integer(df$Col2), x = df$Coldata)`<br>
`colnames(sparseMat) = levels(df$Col2)`<br>
`rownames(sparseMat) = levels(df$Col1)`<br>

### 8) Save cookies information while scraping an url
`library(rvest)`<br>
`library (tidyverse)`<br>
`library (httr)`<br>

`#retrieve session cookie by taking the url of the main page`<br>
`url <- "https://salesweb.civilview.com/Sales/SalesSearch?countyId=3" #sample url`<br>
`urlInfo <- GET(url)`<br>

`#create a new dataframe to save the data (called df)`<br>
`see tip 3 of the cheatsheet`<br>
`#retrieve the data (example for the first row data)`<br>
`responseDetail <- GET(df[1,c('Details')], set_cookies(`urlInfo$cookies[6]` = paste0('"',urlInfo$cookies[7],'"')))`<br>

`#scrape the html tags of interest`<br>
`readUrlHtmlDetail <- read_html(responseDetail) %>% html_nodes("td")`<br>
  
 [Here's](https://stackoverflow.com/questions/55169844/unable-to-connect-to-https-site-with-r/55346855#55346855) a working sample.
  
### 9) Nice looking tables in R Markdown
 
Follow this [complete guide](https://cran.r-project.org/web/packages/kableExtra/vignettes/awesome_table_in_html.html) by <i>Hao Zhu</i> on using the `kableExtra` package.

### 10) Correlation plot
`library(ISLR)`<br>
`library(GGally)`<br>
`ggpairs(iris)`<br>
 
### 11) Using <i>tapply</i> function
`tapply(argument 1, argument 2 , argument 3)`<br>
tapply splits (groups) the data by the second argument you give, and then applies the third argument function to the variable given as the first argument.<br>

### 12) Working with NAs

Remove NAs values: `na.omit(...)`
  
  
  
