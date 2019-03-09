# R-Cheatsheet

Often I'm googling for the very same coding issues! Think that it would be a good idea to start my own R-cheatsheet. :grin:

### R Markdown coding character

`

### Change value of a categorical variable in a dataframe

`levels(df$column)[levels(df$column)=='Old Value'] <- 'New Value'`

### Create a dataframe

`column1 <- c(1,2,3,4,5)`

`column2 <- c('hi', 'have', 'a', 'nice', 'day')`

`column3 <- c(TRUE, TRUE, FALSE, TRUE, FALSE)`

`df <- data.frame(column1, column2, column3)`
