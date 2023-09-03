# Performing Feature Engineering in Customer Analytics
# Project/Business Problem: Customer Analytics from an online Music Merchant Company(aka iTunes)

# Business Questions: Learning about Customers;
# - What products are customers purchasing?
# - which Customers are similar?
# - what Relates to Customers Re-purchasing within 90-Days ie who is most likely or least likely to purchase in next 90-days?


# We are going to:
# - Wrangle d data from d database
# - Feature Engineer the data
# - Model using XGBoost model


#  Learning Path:
# - Found Relationships between Customers (using step_umap function)
# - Combined many tables within a Customer Database
# - Used Reduce Dimensionality (eg Number of Zip Codes) with Hashing (step_feature_hash)
# - Modelled 90-Day Re-purchases with XGBOOST
# - Identified Important Features for Business Intelligence


# install.packages("textrecipes")
# install.packages("embed")

library(DBI)
library(RSQLite)
library(plotly)
library(skimr)
library(GGally)
library(tidymodels)
library(embed)      # contains step_umap() and step_feature_hash()
library(textrecipes)
library(vip)
library(lubridate)
library(tidyverse)


# Database
# - Database connection: set your working directory to d folder containing d database file
con <- DBI::dbConnect(SQLite(), "Chinook_Sqlite.sqlite")
con

# - Database Tables
dbListTables(con)  # view all name of all d tables in d database file u imported

tbl(con, "Invoice")  # lets view d data in Invoice table

tbl(con, "Invoice") %>% collect()  # using collect() will make/turn d data in Invoice table into a tibble/dataframe


#---> Quick Tip: How to view all d tables and their data from d Database ----
dbListTables(con) %>% map(~ tbl(con, .))


#-----> Lets View various Tables in d Database ------>

# - Invoices Table
invoices_tbl <- tbl(con, "Invoice") %>% collect()
invoices_tbl

invoices_tbl <- invoices_tbl %>%
  mutate(InvoiceDate = as_date(InvoiceDate))

invoices_tbl %>% glimpse()

invoices_tbl %>% write_rds("invoices_tbl.rds")


# - Customers Table
customers_tbl <- tbl(con, "Customer") %>% collect()

customers_tbl %>% glimpse()


# Joining/Combining data from diff tables
invoice_lines_tbl <- tbl(con, "InvoiceLine") %>%   # Invoice Lines Table
  left_join(tbl(con, "Track") %>%    #  Track Table
              select(-UnitPrice) %>%  # removes d stated column
              rename(TrackName = Name),
            by = "TrackId") %>%
  left_join(tbl(con, "Genre") %>%    # Genre Table
              rename(GenreName = Name),
            by = "GenreId") %>%
  left_join(tbl(con, "Album") %>%      # Album Table
              rename(AlbumTitle = Title),
            by = "AlbumId") %>%
  left_join(tbl(con, "Artist") %>%     # Artist Table
              rename(ArtistName = Name),
            by = "ArtistId") %>%
  left_join(tbl(con, "Invoice") %>%   # Invoice Table
              select(InvoiceId, CustomerId),
            by = "InvoiceId") %>%
  select(-ends_with("Id"), starts_with("Invoice"), starts_with("Customer")) %>%  # removes any column that ends with 'Id'
  relocate(contains("Id"), .before = 1) %>%
  collect()     # turns all collated data into tibble/dataframe

invoice_lines_tbl %>% glimpse()

invoice_lines_tbl %>% write_rds("invoice_lines_tbl.rds")

invoice_lines_tbl <- read_rds("invoice_lines_tbl.rds")

# Check Dataset
invoice_lines_tbl %>% skim()

# Close Database Connection
DBI::dbDisconnect(con)




#-----> Feature Engineering on Customer Analytics
# 1.0 Product Relationship: Customer-Artist
# Here we look the Customers and d Artist they love buying from
# - Focus: Invoice Lines
invoice_lines_tbl %>% distinct(ArtistName)

# Pivot Wider (Dummy variables)
customers_artists_tbl <- invoice_lines_tbl %>%
  select(CustomerId, ArtistName) %>%
  count(CustomerId, ArtistName) %>%
  pivot_wider(names_from = ArtistName, values_from = n,
              values_fill = 0, names_prefix = "artist_",  # adds d stated prefix to d names of new columns
              names_sep = "_")

customers_artists_tbl

# Based on d fact that Recipes deals with only numerical values, we have to use pivot_wider() to perform One-Hot Encoding on categorical columns


# Dimensionality Reduction Technique using UMAP
# This involves reducing/compressing/condensing d many columns/dimensions/variables in d data into few columns/dimensions/variable
# UMAP is similar to PCA. They are both Dimensionality Reduction Techniques.
recipe_spec_umap <- recipe(~ ., customers_artists_tbl) %>%
  step_umap(-CustomerId, num_comp = 20,         # step_umap() will condense the columns(due to the pivot_wider used earlier we have 165 columns) into 20 columns
            keep_original_cols = FALSE, seed = c(123, 123))

customers_artists_umap_tbl <- recipe_spec_umap %>% prep() %>% juice()
customers_artists_umap_tbl   # dis creates new columns with the names umap_01, umap_02, umap_03...... till umap_20

customers_artists_umap_tbl %>% write_rds("customers_artists_umap_tbl.rds")

customers_artists_umap_tbl <- read_rds("customers_artists_umap_tbl.rds")


# Question: Which Customers are Buying from similar Artists
# - creating a 2D plot
g <- customers_artists_umap_tbl %>% 
  ggplot(aes(umap_01, umap_02)) +    # using d stated columns
  geom_point(aes(text = CustomerId), alpha = 0.5)

ggplotly(g)


# - Creating 3D plot using Plotly that u can pan around
customers_artists_umap_tbl %>%
  plot_ly(x = ~umap_01, y = ~umap_02, z = ~umap_03,
          color = umap_04, text = ~CustomerId) %>%
  add_markers()


# lets investigate Customers 35, 55 and 16 since they are close to each other and are the same color in d 3D plot
invoice_lines_tbl %>%
  filter(CustomerId %in% c(35, 55, 16)) %>%
  count(CustomerId, ArtistName) %>%
  group_by(CustomerId) %>%
  arrange(-n, .by_group = TRUE) %>%  # sort in descending order by grouped/partitioned rows
  slice(1:5)

invoice_lines_tbl %>% glimpse()


# Aggregation Features: Length of Song
# Lets see if length of songs have an impact on Customers preference ie if some customers love long-duration songs than short-duration song
customer_song_len_tbl <- invoice_lines_tbl %>%
  select(CustomerId, Milliseconds) %>%  #  Milliseconds column has length of songs data
  group_by(CustomerId) %>%
  summarise(enframe(quantile(Milliseconds, probs = c(0, 0.25, 0.5, 0.75, 1)))) %>%  # creates a new column displaying d min length(0), max length(1), median length(0.5), 25th percentile length(0.25), 75th(0.75) percentile length of songs listened to by each customer. We can see that Customer 1 listens to long-duration songs the most  
  ungroup() %>%
  mutate(name = str_remove_all(name,"%")) %>% 
  pivot_wider(names_from = name, values_from = value,  
              names_prefix = "song_len_q")  # adds dis prefix to the names of d new columns
# we use pivot_wider() to display d length of songs listened to by each distinct customer(ie CustomerId by song length)

customer_song_len_tbl %>%
  arrange(-song_len_q100)  # sort the stated column in descending order
# We can deduce that Customer 51 and 40 listen to longest-duration songs since they have d highest values in song_len_q100 column(whih is max length of songs)



# 2.0 Purchase Relationships: Using Date Features and Price Features to know Customer Purchases
# - Focus: Invoices Table
# Date and Price Features
max_date <- max(invoices_tbl$InvoiceDate)  # displays d max date value(which is d most recent/last date) in Invoicedate column
max_date

customer_invoice_tbl <- invoices_tbl %>%
  select(CustomerId, InvoiceDate, Total) %>%
  group_by(CustomerId) %>%
  summarise(
    # Date Features
    inv_most_recent_purchase = (max(InvoiceDate) - max_date) / ddays(1),  # shows how long it has been (in days) since d most recent date of purchase made by each customer (ie the max date/last date of purchase of that customerin d date column - max date/last date of the date column divided by duration in days). dday(1) means add 1 day
    inv_tenure = (min(InvoiceDate) - max_date) / ddays(1),   # shows how long it has been (in days) since each customer made their 1st purchase so as to know how long they have been customers (ie the min date/1st date of purchase of that customer in d date column - max date/last date of the date column divided by duration in days)
    
    # Prices/Quantity features
    inv_count = n(),
    inv_sum = sum(Total, na.rm = TRUE),
    inv_avg = mean(Total, na.rm = TRUE)
  )

customer_invoice_tbl
# We can see that Customer 1 most recent purchase was 137 days ago(from d last date in Invoicedate column) and he made his 1st purchase 1382 days ago (from d last date/max date in Invoicedate column)

customer_invoice_tbl %>%
  ggpairs(columns = 2:ncol(.), 
          title = "Customer Aggregated Invoice Features")



# 3.0 Customer Features
# Focus: Customers Table
customers_tbl %>% skim()

# Joining
customers_joined_tbl <- customers_tbl %>%
  select(contains("Id"), PostalCode, Country, City) %>%
  left_join(customer_invoice_tbl, by = "CustomerId") %>%
  left_join(customer_song_len_tbl, by = "CustomerId") %>%
  left_join(customers_artists_umap_tbl, by = "CustomerId") %>%
  rename_at(.vars = vars(starts_with("umap")), .funs = ~str_glue("artist_{.}"))  # add artist_ as prefix name to any column that starts with umap

customers_joined_tbl %>% glimpse()

customers_joined_tbl %>% skim()

customers_joined_tbl %>% write_rds("customers_joined_tbl.rds")

customers_joined_tbl <- read_rds("customers_joined_tbl.rds")


# C. Modeling Setup
# Create/Make a new feature/column called Target since we wanto which customer is most likely or least likely to purchase in next 90-days?
full_data_tbl <- customers_joined_tbl %>%
  mutate(Target = ifelse(inv_most_recent_purchase >= -90, 1, 0)) %>% 
  mutate(Target = as.factor(Target)) %>%
  select(-inv_most_recent_purchase) %>%
  relocate(Target, .after = CustomerId)


# Data Splitting
set.seed(123)

splits <- initial_split(full_data_tbl, prop = 0.80)

write_rds(splits, "masplits.rds")

splits <- read_rds("masplits.rds")

# Recipe Embedding: Dummy
recipe_spec_dummy <- recipe(Target ~ ., training(splits)) %>%
  add_role(CustomerId, new_role = "Id") %>%
  step_knnimpute(PostalCode, impute_with = vars(Country, City)) %>%   # we have missing value but imputeing them is optional becuz XGBoost(which is d ML model we are going to use) can handle NAs. can use step_bagimpute() instead of step_knnimpute()
  step_dummy(Country, City, PostalCode, one_hot = TRUE)  # performing One Hot Encoding on d stated columns

recipe_spec_dummy %>% prep() %>% juice()
# The One Hot Encoding performed on those stated columns will result in many additional columns(over 130 columns) due to the recipes and we do not want that if uar going to do a Regression model cause it will mk d model to over-fit,
# But we are doing XGBoost model so its OK. But lets just use num_hash parameter in step_feature_hash() on those stated columns so as to condense d many columns that will be generated into a lower number of columns. 
# step_feature_hash() is very similar to UMAP operation used much earlier to condense columns.


# Recipe Embedding: Hash
recipe_spec_hash <- recipe(Target ~ ., training(splits)) %>%
  add_role(CustomerId, new_role = "Id") %>%
  # step_knnimpute(PostalCode, impute_with = vars(Country, City)) %>%    # since XGBoost model can handle NAs lets not impute the NAs
  step_feature_hash(Country, City, PostalCode, num_hash = 15)  # using num_hash parameter in step_feature_hash() on the stated columns

recipe_spec_hash %>% prep() %>% juice()


#----> XGBoost Modeling using Workflow
# - Using Recipes Embedding: Dummy
wflw_fit_xgb_dum <- workflow() %>%
  add_model(spec = boost_tree(mode = "classification") %>%
              set_engine("xgboost")) %>%
  add_recipe(recipe_spec_dummy) %>%
  fit(training(splits))

bind_cols(wflw_fit_xgb_dum %>% predict(testing(splits), type = "prob"),
          testing(splits)) %>%
  yardstick::roc_auc(Target, .pred_1)
# roc_auc value should be closer to 1 for a legit model


# - Using Recipes Embedding: Hash 
wflw_fit_xgb_hash <- workflow() %>%
  add_model(spec = boost_tree(mode = "classification") %>%
              set_engine("xgboost")) %>%
  add_recipe(recipe_spec_hash) %>%
  fit(training(splits))

bind_cols(wflw_fit_xgb_hash %>% predict(testing(splits), type = "prob"),
          testing(splits)) %>%
  yardstick::roc_auc(Target, .pred_1)
# roc_auc value should be closer to 1 for a legit model. Its worse than XGBoost using Recipes Embedding: Dummy value above


# Feature Importance
wflw_fit_xgb_hash$fit$fit$fit %>% vip()
# Since inv_tenure,  song_len_q100 and song_len_q50 columns are most important features to d model (as they determine to a great extent if customers will repurchase in 90-days), lets analyse dem individually

# Analysing the Distribution in inv_tenure column
full_data_tbl %>%
  ggplot(aes(inv_tenure, fill = Target)) +
  geom_density(alpha = 0.5)
# based on d legend bar, 1 is more likely to repurchase in 90-days while 0 is more likely not to repurchase in 90-days 

full_data_tbl$inv_tenure %>% range()


# Analysing the Distribution in song_len_q50 column
full_data_tbl %>%
  ggplot(aes(song_len_q50, fill = Target)) +
  geom_density(alpha = 0.5)


# Analysing the Distribution in artist_umap_16 column
full_data_tbl %>%
  ggplot(aes(artist_umap_16, fill = Target)) +
  geom_density(alpha = 0.5)
# we need to still analyse dis artist_umap_16 column to know which artist it is



# Make Full Predictions
bind_cols(wflw_fit_xgb_hash %>% predict(full_data_tbl, type = "prob"),
          customers_joined_tbl) %>%
  write_rds("customer_predictions_tbl.rds")


#----> Conclusions And Recommendations
# - Found relationships between Customers (step_umap)
# - Combined many tables within a Customer Database
# - Used Reduce Dimensionality (e.g Number of Zip Codes) with Hashing (step_feature_hash)  
# - Modeled 90-Day Re-purchases with XGBoost
# - Identified Important features for Business intelligence
  
  



