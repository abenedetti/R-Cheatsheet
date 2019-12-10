# R-Cheatsheet

Often I'm googling for the very same coding issues! Think that it would be a good idea to start my own R-cheatsheet. :grin:

### 1) R Markdown special characters
backtick: ` (ALT + 9 6)<br>
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

[Using apply, sapply, lapply in R](https://www.r-bloggers.com/using-apply-sapply-lapply-in-r/)<br>

### 12) Working with NAs

Remove NAs values: `na.omit(...)`

### 13) Splitting datasets (train/test sets)

#### for continuous outcome (70% train data)

`spl = sample(1:nrow(data), size=0.7 * nrow(data))`<br>
`train = data[spl,]`<br>
`test = data[-spl,]`<br>

#### for categorical outcome (70% train data)

`library(caTools)`<br>
`spl = sample.split(data$Outcome, SplitRatio = 0.7)`<br>
`train = subset(data, spl == TRUE)`<br>
`test = subset(data, spl == FALSE)`<br>
  
### 14) Create ROC curves (receiver operating characteristic)

`library(ROCR)`<br>
`# Prediction function`<br>
`ROCRpred = prediction(`predictedValues`,` actualValues`)`<sup>(*)</sup><br>
`# Performance function`<br>
`ROCRperf = performance(ROCRpred, "tpr", "fpr")`<br>
`# Plot ROC curve`<br>
`plot(ROCRperf)`<br>
`# Add colors`<br>
`plot(ROCRperf, colorize=TRUE)`<br>
`# Add threshold labels`<br>
`plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))`<br>
`# Calculate the AUC (area under the curve) of the model`<br>
`as.numeric(performance(ROCRpred, "auc")@y.values)`
  
 <sup>(*): for CART the predicted probabilities of the test can be obtained by removing the type="class" argument when making predictions, and taking the second column of the resulting object.</sup>
  
### 15) Predictions with `lm` and `glm` packages

Given a `train` and `test` sets, the dependent variable `outcome` and independent variables `x`, `y` and `z`.

* for linear regression:
  model: `aModel <- lm(outcome ~ x + y + z, data = train)`<br>
  prediction on test set <sup>(*)</sup>: `prediction <- predict(aModel, newdata = test$outcome)`<br>

* for logistic regression:
  model <sup>(*)</sup>: `aLogModel <- glm(outcome ~ x + y + z, data = train, family = binomial)`<br>
  prediction on test set <sup>(**)</sup>: `predictionLog <- predict(aLogModel, type="response", newdata = test)`<br>
  confusion matrix with threshold of 0.5: `table(test$outcome, predictionLog > 0.5)`<br>
  test set AUC with ROCR library: `ROCRpred <- prediction(predictionLog, test$outcome)`<br>
  `as.numeric(performance(ROCRpred, "auc")@y.values)`<br>
  
  <sup>(*): interaction terms between categorical variables can be added by using the colon. In this model `lm(outcome ~ x + y + z + x:y, data = train)` we added `x:y` interaction term to consider the case when x=1 AND y=1</sup><br>
  <sup>(**): the `newdata` argument is filled when when want to make a prediction on a dataset that wasn't used to train our model</sup>
  
  
### 16) Missing data imputation with *Multiple Imputation by Chained Equations*
  
 * load `mice` package<br>
 * suppose we have dataframe `aDataFrame` with X, Y, Z and outcome variables<br>
 * suppose Y and Z have a decent number of NAs (missing values)<br>
 * create a new dataframe `simple` with X, Y, Z (we don't want the `outcome` variable): `simple <- aDataFrame[c("X","Y","Z")]`<br>
 * create a new dataframe `imputed`: `imputed <- complete(mice(simple))`<br>
 * by checking `summary(imputed)` we see that there are no more NAs <br>
 * last step is to copy over the new values of imputed dataframe: `aDataframe$Y <- imputed$Y` and `aDataframe$Z <- imputed$Y`


### 17) Classification and regression trees CART

`library(rpart) #main library`<br>
`library(rpart.plot) #plots library`<br>

Given a `train` and `test` sets, the dependent variable `outcome` and independent variables `x`, `y` and `z`.

#### Classification (categorical `outcome` variable)

* tree model <sup>(**)</sup>:`Tree = rpart(outcome ~ x + y + z, data = train, method="class", minbucket=25)`
* tree plot: `prp(Tree)`
* prediction <sup>(*)</sup>: `predict(Tree, newdata = test, type = "class")`
* accuracy of the model can be assessed through a confusion matrix

#### Regression (continuous `outcome` variable)

* tree model <sup>(**)</sup>:`Tree = rpart(outcome ~ x + y + z, data = train)`
* tree plot: `prp(Tree)`
* prediction <sup>(*)</sup>: `predict(Tree, newdata = test)`
* accuracy of the model can be assessed through the SSE calculation


<sup>(*): the `type="class"` is used to create predicted values and not probabilities, it is equivalent to a threshold of 0.5</sup><br>
<sup>(**): we can use other parameters to create the model instead of `minbucket`. One is by using the `cp` parameter, which optimal value can be determined by a CV method - see point 19 below - and/or add a penalty matrix to "balance" outcomes in multiclass classification: `Tree = rpart(outcome ~ x + y + z, data = train, method="class", parms=list(loss=PenaltyMatrix))` where `PenaltyMatrix` is a penalty matrix.</sup>

### 18) Random forests

`library(randomForest)`

Given a `train` and `test` sets, the dependent variable `outcome` and independent variables `x`, `y` and `z`.

`modelRF <- randomForest(outcome ~ x + y + z, data = train, ntree=<number of trees>, nodesize=<eqv to minbuckey value>)`<br>
`predictRF <- predict(modelRF, newdata = test)`<sup>(*)</sup><br>

<sup>(*): you can obtain probabilities for a random forest by adding the argument `type="prob"`</sup><br>

### 19) Cross validation

`library(caret)`<br>
`library(e1071)`<br>

Given a `train` and `test` sets, the dependent variable `outcome` and independent variables `x`, `y` and `z`.

*Define cross-validation experiment*<br>
`numFolds = trainControl( method = "cv", number = 10 )`<br>
`cpGrid = expand.grid( .cp = seq(0.01,0.5,0.01))`

Cross validation of 10 folds `number` and complexity parameter canditate parameters in range 0,01 to 0,5 by 0,01 steps.

*Perform the cross validation*<br>
`train(outcome ~ x + y + z, data = train, method = "rpart", trControl = numFolds, tuneGrid = cpGrid )`

The cross validation returns the output cp table with the optimal cp value (cp<sub>opt</sub>)

*Create a new CART model with cp parameter instead of minbucket*<br>
`modelCV = rpart(outcome ~ x + y + z, data = train, method="class", cp = `cp<sub>opt</sub>`)`<br>
<sub>we set the method classification since we're dealing with a classification problem</sub>

*Make predictions*<br>
`PredictCV = predict(StevensTreeCV, newdata = Test, type = "class")`<br>
<sub>we set the method classification since we're dealing with a classification problem</sub>

### 20) Remove variable while building a model

Given a `train` and `test` sets, the dependent variable `outcome` and independent variables `x`, `y` and `z`.

* we can remove a single variable with the `-`: `(...)outcome ~ . - y(...)`
* we can remove a list of variable with:<br>
  `nonvars = c("x", "z")`<br>
  `trainPartial = train[ , !(names(train) %in% nonvars) ]`<br>
  `testPartial = test[ , !(names(test) %in% nonvars) ]`<br>
  `(...)outcome ~ ., data=trainPartial(...)`<br>

### 21) Natural language pre processing

Pre processing done by using the Bag of Words methods follows those steps in R:

`df = read.csv("a_file.csv", stringsAsFactors=FALSE)`<br>
`library(tm)`<br>
`library(SnowballC)`<br>
`# create corpus`<br>
`corpus = VCorpus(VectorSource(df$text_data_column))`<br>
`# convert to lower-case`<br>
`corpus = tm_map(corpus, content_transformer(tolower))`<br>
`# remove punctuation`<br>
`corpus = tm_map(corpus, removePunctuation)`<br>
`# remove stopwords and/or other words`<br>
`corpus = tm_map(corpus, removeWords, c("a_word_to_remove", stopwords("english")))`<br>
`# stem document`<br>
`corpus = tm_map(corpus, stemDocument)`<br>
`# look at the corpus content`<br>
`corpus[[1]]$content`<br>

`# create matrix that contains the number of occurrence of each word`<br>
`frequencies = DocumentTermMatrix(corpus)`<br>
`# look at matrix `<br>
`inspect(frequencies[1000:1005,505:515])`<br>
`# check for sparsity`<br>
`findFreqTerms(frequencies, lowfreq=20)`<br>
`# remove sparse terms`<br>
`sparse = removeSparseTerms(frequencies, 0.97)` <sub>we'll remove any term that doesn't appear in at least 3% of the documents</sub><br>
`# convert to a data frame`<br>
`dfSparse = as.data.frame(as.matrix(sparse))`<br>
`# make all variable names R-friendly`<br>
`colnames(dfSparse) = make.names(colnames(dfSparse))`<br>

### 22) Create a dataframe column based on conditions

`df$newColumn = ifelse(grepl(stringToSearch,df$ColumnSearchWithin,fixed=TRUE), newValueIfTrue, newValueIfFalse)`

### 23) Hierarchical clustering

* compute euclidean distance: `distances = dist(df, method = "euclidean")`<br>
* create the hierarchical clustering: `clusterDf <- hclust(distances, method = "ward.D")`<br>
* plot the dendrogram: `plot(clusterDf)`<br>
* plot the clusters: `rect.hclust(clusterDf, k = <num of selected clusters>, border = "red")`
* assign points to clusters: `clusterGroups = cutree(clusterDf, k = <num of clusters>)`<br>
* visualize the cluster:<br>
`dim(clusterDf) = c(nrow(dfMatrix), ncol(dfMatrix))`<br>
`image(clusterDf, axes=FALSE, col=grey(seq(0,1,length=256)))`<br>

### 24) k-Means clustering

Let's have a train data set:<br>
`dataTrainMatrix = as.matrix(dataTrain)`<br>
`dataTrainVector = as.vector(dataTrainMatrix)`<br>

* specify number of clusters: `k = 5`<br>
* run k-means: `KMC = kmeans(dataTrainVector, centers = k, iter.max = 1000)<br>
* view cluster data: `str(KMC)`<br>

* apply the k-means clustering results (train) to a test set:<br>
`library(flexclust)`<br>
`KMC.kcca = as.kcca(KMC, dataTrainVector)`<br>
`dataTestClusters = predict(KMC.kcca, newdata = dataTestVector)`<br>

* visualize the cluster:<br>
`dim(dataTestClusters) = c(nrow(dataTestMatrix), ncol(dataTestMatrix))`<br>
`image(dataTestClusters, axes=FALSE, col=rainbow(k))`<br>

### 25) Clustering scree plots

A standard scree plot has the number of clusters on the x-axis, and the sum of the within-cluster sum of squares on the y-axis. The within-cluster sum of squares for a cluster is the sum, across all points in the cluster, of the squared distance between each point and the centroid of the cluster.  We ideally want very small within-cluster sum of squares, since this means that the points are all very close to their centroid.<br>

* define the range of clusters: `NumClusters = seq(2,10,1)`<br>
* define within-cluster sum of squares: `SumWithinss = sapply(2:10, function(x) sum(kmeans(dfVector, centers=x, iter.max=1000)$withinss))`<br>
* plot the scree plot: `plot(NumClusters, SumWithinss, type="b")`<br>

To determine the best number of clusters using this plot, we want to look for a bend, or elbow, in the plot. This means that we want to find the number of clusters for which increasing the number of clusters further does not significantly help to reduce the within-cluster sum of squares.

### 26) Normalize dataframe variables (preprocess)

Standardization/normalization is a procedure that leads to a random variable distributed according to an average μ and variance σ<sup>2</sup>, to a random variable with a "standard" distribution, that is to say zero mean and variance equal to 1.<br>

`library(caret)`<br>
`preproc = preProcess(dfRaw)`<br>
`dfNorm = predict(preproc, dfRaw)`<br>

### 27) Working with dates

Format text to date: `strptime(df$dateInText, format = "%m/%d/%y %H:%M")`


### 28) Ggplot2 basics

* ggplot code template:<br>
`ggplot(data = <DATA>) +`<br> 
`  <GEOM_FUNCTION>(`<br>
`     mapping = aes(<MAPPINGS>),`<br>
`     stat = <STAT>,`<br>
`     position = <POSITION>`<br>
`  ) +`<br>
`  <COORDINATE_FUNCTION> +`<br>
`  <FACET_FUNCTION>`<br>

* line plot: `ggplot(<df>, aes(x=<x>, y=<y>)) + geom_line(<aes(group=1)>) + xlab(<title>) + ylab(<title>)`<br>

* heatmap: `ggplot(<df>, aes(x=<x>, y=<y>)) + geom_tile(<aes(fill=<FillVar>)>) + scale_fill_gradient(name=<legend name>, low=<lowCol>, high=<highCol>) + theme(axis.title.y = element_blank())`<br>

* maps:<br>
`library(maps)`<br>
`library(ggmap)`<br>
`# Load a map of Chicago into R:`<br>
`chicago = get_map(location = "chicago", zoom = 11)`<br>
`# Look at the map`<br>
`ggmap(chicago)`<br>
`# Plot on the map:`<br>
`ggmap(chicago) + geom_point(data = df, aes(x = Longitude, y = Latitude))`<br>

* state map:<br>
`library(ggplot2)`<br>
`library(maps)`<br>
`library(ggmap)`<br>
`#draw us states map`<br>
`statesMap <- map_data("state")`<br>
`ggplot(statesMap, aes(x = long, y = lat, group = group)) + geom_polygon(fill = "white", color = "black")`<br>
<sub>The structure of the state contains a variable named *group*, that defines the different shapes or polygons on the map. Sometimes a state may have multiple groups, for example, if it includes islands. The variable *order* defines the order to connect the points within each *group*, and the variable *region* gives the name of the state.</sub>

### 29) Convert factor variable to numeric

`df$NumVar <- as.numeric(as.character(df$FactVar))`

### 30) Working with igraph

`library(igraph)`<br>
`g = graph.data.frame(d, FALSE, vertices) #load a graph object`<br>
`plot(g, vertex.size=5, vertex.label=NA) #plot a graph by setting the vertices size`<br>
`degree(g) #retrieve the graph degree`<br>
`V(g)$size V(g)$color #set graph properties on size & color`<br>

### 30) Working with wordcloud

`library(wordcloud)`<br>
`wordcloud(<words>,<words frequencies>, scale = c(2, 0.25))`



