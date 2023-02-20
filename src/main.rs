use std::cmp::Ordering;
use std::path::Path;
use std::fs::File;
use std::convert::TryFrom;

use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::arrays::{MutArray, Array2, Array};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::{train_test_split, cross_validate, KFold};
use smartcore::ensemble::random_forest_classifier::*;

use polars::prelude::*;


#[derive(Debug)]
pub struct BestParameters {
    pub n_trees: u16,
    pub max_depth: u16,
    pub max_features: usize,
}

impl BestParameters {
    pub fn new(n_trees: u16, max_depth: u16, max_features: usize) -> Self {
        BestParameters {
            n_trees,
            max_depth,
            max_features,
        }
    }
}

/// Read a CSV with headers.
pub fn read_csv<P: AsRef<Path>>(path: P) -> PolarsResult<DataFrame> {
    let file = File::open(path).expect("Cannot open file.");

    CsvReader::new(file)
        .has_header(true)
        .finish()
}

/// Split a polar DataFrame into features DataFrame and target DataFrame.
pub fn feature_and_target(df: &mut DataFrame) -> (PolarsResult<DataFrame>, PolarsResult<DataFrame>) {
    let features = df.select(vec![
        "1stD_offense", "TotYd_offense", "PassY_offense", "RushY_offense",
        "TO_offense", "1stD_defense", "TotYd_defense", "PassY_defense",
        "RushY_defense", "TO_defense"
    ]);

    // Converts "W" to 1, "L" and "T" to 0
    fn str_to_u32(str_val: &Series) -> Series {
        str_val.utf8()
            .unwrap()
            .into_iter()
            .map(|opt_name: Option<&str>| {
                opt_name.map(|name: &str| {
                    u32::from(name == "W")
                })
             })
            .collect::<UInt32Chunked>()
            .into_series()
    }
    
    df.apply("result", str_to_u32).unwrap();

    let target = df.select(vec!["result"]);

    (features, target)
    
}

/// Convert features DataFrame to a smartcore DenseMatrix.
pub fn convert_features_to_matrix(df: &DataFrame) -> Result<DenseMatrix<f64>, PolarsError>{
    let nrows = df.height();
    let ncols = df.width();
    
    let features_res = df.to_ndarray::<Float64Type>().unwrap();
    // create a zero matrix and populate with features
    let mut x_matrix: DenseMatrix<f64> = Array2::zeros(nrows, ncols);
    // populate the matrix 
    // initialize row and column counters
    let mut col:  u32 = 0;
    let mut row:  u32 = 0;

    for val in features_res.iter(){
        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        
        x_matrix.set((m_row, m_col), *val);
        // check what we have to update
        if m_col == ncols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    Ok(x_matrix)
}

/// Perform grid search parameter tuning with cross validation and returns the "best" parameters.
/// 
/// This function performs a grid search on the following parameters:
/// - Number of trees
/// - The depth of each tree
/// - The max number of features to consider when splitting a node
fn grid_search(x_train: &DenseMatrix<f64>, y_train: &Vec<u32>) -> Option<BestParameters> {
    let mut best_score = 0.;
    let mut best_params: Option<BestParameters> = None;

    let num_features = x_train.shape().1;
    let mtry = (num_features as f64).sqrt().floor() as usize;

    let n_trees = vec![25, 50, 75, 100];
    let max_depth = vec![6, 8, 10, 12, 14, 50];
    let max_features = vec![(mtry / 2) as usize, mtry, (mtry * 2) as usize];

    for n_tree in n_trees {
        for m_depth in &max_depth {
            for m_feat in &max_features {
                println!("Running {} trees of depth {} with {} features", n_tree, m_depth, m_feat);

                let cv_score = cross_validate(
                    RandomForestClassifier::new(),
                    x_train,
                    y_train,
                    RandomForestClassifierParameters::default()
                        .with_n_trees(n_tree)
                        .with_max_depth(*m_depth)
                        .with_m(*m_feat),
                &KFold::default().with_n_splits(10),
                &accuracy
                ).unwrap();
                println!("Score: {}", cv_score.mean_test_score());
                
                if cv_score.mean_test_score() > best_score {
                    best_score = cv_score.mean_test_score();
                    best_params = Some(BestParameters {
                        n_trees: n_tree,
                        max_depth: *m_depth,
                        max_features: *m_feat
                    });
                }
            }
        }
    }

    println!("Best score: {}", best_score);
    println!("Best parameters: {:?}", best_params);

    best_params
}

/// Drop a column from a smartcore DenseMatrix.
pub fn drop_column(x: &mut DenseMatrix<f64>, drop_column_index: usize) -> DenseMatrix<f64> {
    let (nrows, ncols) = x.shape();
    let new_ncols = ncols - 1;

    let mut x_matrix: DenseMatrix<f64> = Array2::zeros(nrows, new_ncols);

    for row in 0..nrows {
        for col in 0..ncols {
            // compare the current column index to the index of the column we want to drop
            match col.cmp(&drop_column_index) {
                Ordering::Less => x_matrix.set((row, col), *x.get((row, col))),
                Ordering::Greater => x_matrix.set((row, col - 1), *x.get((row, col))),
                Ordering::Equal => continue,
            }
        }
    }

    x_matrix
}

/// Calculate the variable importance of each feature use percent decrease in accuracy.
pub fn variable_importance(x_train: &DenseMatrix<f64>, y_train: &Vec<u32>, parameters: BestParameters, feature_names: Vec<&str>, accuracy_score: &f64) -> Vec<(String, f64)> {
    let mut importance: Vec<(String, f64)> = Vec::new();

    for (col_index, feature_name) in feature_names.iter().enumerate() {
        let mut new_train = x_train.clone();
        new_train = drop_column(&mut new_train, col_index);

        let classifier = RandomForestClassifier::fit(
            &new_train,
            y_train,
            RandomForestClassifierParameters::default()
                .with_n_trees(parameters.n_trees)
                .with_max_depth(parameters.max_depth)
                .with_m(parameters.max_features)
        ).unwrap();

        let new_accuracy_score = accuracy(y_train, &classifier.predict(&new_train).unwrap());

        let accuracy_decrease = ((accuracy_score - new_accuracy_score) / accuracy_score) * 100.0;

        importance.push((feature_name.to_string(), accuracy_decrease));
    }

    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    importance
}

fn main() {
    // Read input data 
    let ifile = "data/season_2021.csv";
    let mut df = read_csv(ifile).unwrap();
    
    
    let (features, target) = feature_and_target(&mut df);
    let x_matrix = convert_features_to_matrix(features.as_ref().unwrap());

    // println!("{}", target.unwrap().head(Some(10)));

    let target_array = target.unwrap().to_ndarray::<UInt32Type>().unwrap();

    let mut y: Vec<u32> = Vec::new();
    for val in target_array.iter(){
        y.push(*val);
    }

    let (x_train, x_test, y_train, y_test) = train_test_split(
        &x_matrix.unwrap(),
        &y, 
        0.3,
        true,
        Some(42)
    );

    let classifier = RandomForestClassifier::fit(&x_train, &y_train, Default::default()).unwrap();

    let y_hat = classifier.predict(&x_test).unwrap();

    println!("Accuracy of basic model: {}", accuracy(&y_test, &y_hat));
    // 78.95% accuracy

    // println!("{:?}", RandomForestClassifierParameters::default())

    let best_params = grid_search(&x_train, &y_train).unwrap();
    // Number of trees: 25
    // Max depth: 8
    // Max features: 6
    // 78.18% mean accuracy (average accuracy across each fold)

    // let best_params = BestParameters {
    //     n_trees: 25,
    //     max_depth: 8,
    //     max_features: 6
    // };

    let tuned_classifier = RandomForestClassifier::fit(&x_train, &y_train, RandomForestClassifierParameters::default()
        .with_n_trees(best_params.n_trees)
        .with_max_depth(best_params.max_depth)
        .with_m(best_params.max_features)
    ).unwrap();

    let accuracy_score = accuracy(&y_test, &tuned_classifier.predict(&x_test).unwrap());

    println!("------------Variable Importance------------");
    let variable_importance = variable_importance(&x_train, &y_train, best_params, features.unwrap().get_column_names(), &accuracy_score);
    for (name, importance) in variable_importance.iter() {
        println!("{}: {:.2}%", name, importance);
    }
    println!("-------------------------------------------");

    let y_hat = tuned_classifier.predict(&x_test).unwrap();

    println!("Accuracy of tuned model: {}", accuracy(&y_test, &y_hat));
    // 77.78% accuracy

    let panthers_file = "data/season_2021.csv";
    let mut panthers_df = read_csv(panthers_file).unwrap();
    
    
    let (panthers_features, panthers_target) = feature_and_target(&mut panthers_df);
    let panthers_x_matrix = convert_features_to_matrix(panthers_features.as_ref().unwrap()).unwrap();

    // println!("{}", target.unwrap().head(Some(10)));

    let panthers_target_array = panthers_target.unwrap().to_ndarray::<UInt32Type>().unwrap();

    let mut panthers_y: Vec<u32> = Vec::new();
    for val in panthers_target_array.iter(){
        panthers_y.push(*val);
    }

    let panthers_y_hat = tuned_classifier.predict(&panthers_x_matrix).unwrap();

    println!("Accuracy of predictions on Carolina Panthers 2022 data: {}", accuracy(&panthers_y, &panthers_y_hat));
    // 89.47% accuracy

}
