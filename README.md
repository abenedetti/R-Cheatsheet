# R-Cheatsheet

Often I'm googling for the very same coding issues! Think that it would be a good idea to start my own R-cheatsheet. :grin:

### R Markdown coding character
`
<br>

### Change value of a categorical variable in a dataframe
`levels(df$column)[levels(df$column)=='Old Value'] <- 'New Value'`
<br>

### Create a dataframe
`column1 <- c(1,2,3,4,5)`<br>
`column2 <- c('hi', 'have', 'a', 'nice', 'day')`<br>
`column3 <- c(TRUE, TRUE, FALSE, TRUE, FALSE)`<br>
`df <- data.frame(column1, column2, column3)`
<br>

### Check for R environment infos
`version`<br>
`sessionInfo()`
<br>


