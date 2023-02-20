# nfl-predictor-rust

This project is a spin-off of one that I found utilizing Python and a logistic regression. The ML space in Rust is unfortunately [limited](https://www.arewelearningyet.com/), but I wanted to try `smartcore`'s Random Forest classifier. 

Both datasets are pulled from [Pro Football Reference](https://www.pro-football-reference.com/) using web scraping in Python. I've found some good examples of web scraping in Rust, but haven't had a chance to implement them. Next steps for this project will probably be to add that in so you can set the team and year you want to pull data for.

One of the interesting things about this project, and the `smartcore` crate itself, is the actual default implementation of the Random Forest classifer. Out of the possible parameters that are available to be tuned, I looked at:
- number of trees
- the maximum depth of each tree
- the maximum number of features to use when splitting a node

When I took a look at the surface-level documentation for the tuning parameters, it says that the default for the maximum tree depth is `None`, one of the two possible values for Rust's `Option` type. Having `None` as the default obviously doesn't make much sense when you're talking about how deep a classification tree can be. Digging further, I eventually found that the actual default value is [65'535](https://doc.rust-lang.org/std/u16/constant.MAX.html). 

Things got a little stranger when I tried to fit the classifier while manually setting the some of the default parameters. For some reason, the accuracy I get (78.18%) was about 0.8% lower than when I don't set anything at all (78.95%).

Unfortunately, `smartcore` doens't have a built-in implementation for a grid search or variable importance, so I ended up writing my own using the mean accuracy for tuning and the percent decrease in accuracy for variable importance.

Overall, I had a lot of fun implementing this and definitely learned quite a bit about `smartcore` and the Rust version of the `polars` crate. The classifier ended up having a 89.47% accuracy when predicting using the 2022 season data from the Carolina Panthers.