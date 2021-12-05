pub mod featuretable {
    //! The `featuretable` module implements loading, extracting, and analyzing
    //! representations based on phonological (typically articulatory) features.
    use serde::Deserialize;
    use serde_yaml;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;
    use std::string::String;
    extern crate cached;
    use cached::proc_macro::cached;
    use cached::UnboundCache;


    /// The [`FeatureTable`] struct provides a layer of abstraction over `ft`, a
    /// [`HashMap`] defining the relationship between IPA phonemes and their
    /// featural eqivalents (expressed as vectors of `i8`s). The `fnames` field
    /// provides human-interpretable names for each of the dimensions in the
    /// vectors.
    #[derive(Debug)]
    pub struct FeatureTable {
        pub ft: HashMap<String, Vec<i8>>,
        pub fnames: Vec<String>,
    }

    impl FeatureTable {
        /// Constructs a [`FeatureTable`] from a CSV file at `path`. The CSV
        /// file must have phonemes in the first column and feature
        /// specifications in each of the additional columns (which each column
        /// corresponds to a feature).
        pub fn from_csv(path: &str) -> FeatureTable {
            FeatureTable {
                ft: FeatureTable::read_feature_table(path),
                fnames: FeatureTable::read_feature_names(path),
            }
        }

        /// Constructs a new [`FeatureTable`] using the default data file that
        /// is compiled into the binary.
        pub fn new() -> FeatureTable {
            FeatureTable {
                ft: FeatureTable::default_feature_table(),
                fnames: FeatureTable::default_feature_names(),
            }
        }

        /// Reads a feature table, as a [`HashMap`] from `String`s to `Vec`s,
        /// from a CSV file at a path.
        fn read_feature_table(path: &str) -> HashMap<String, Vec<i8>> {
            let path = Path::new(path);
            let display = path.display();
            let file = match File::open(&path) {
                Err(why) => panic!("Couldn't open {}: {}", display, why),
                Ok(file) => file,
            };
            let mut rdr = csv::Reader::from_reader(file);
            let mut table: HashMap<String, Vec<i8>> = HashMap::new();
            for result in rdr.records() {
                let v = match result {
                    Err(why) => panic!("Bad record: {}", why),
                    Ok(record) => record,
                };
                let k = v[0].to_string();
                let mut tail = Vec::new();
                for i in 1..(v.len()) {
                    tail.push(to_numeric(&v[i]));
                }
                table.insert(k, tail);
            }
            return table;
        }

        /// Reads the names of the features from a CSV file. It is assumed that
        /// they are in the header row in the non-initial columns. 
        fn read_feature_names(path: &str) -> Vec<String> {
            let path = Path::new(path);
            let display = path.display();
            let file = match File::open(&path) {
                Err(why) => panic!("Couldn't open {}: {}", display, why),
                Ok(file) => file,
            };
            let mut rdr = csv::Reader::from_reader(file);
            let headers = match rdr.headers() {
                Ok(result) => result,
                _ => panic!("Cannot read headers from {:?}!", display),
            };
            let mut fnames = Vec::new();
            for name in headers.iter().skip(1) {
                fnames.push(name.to_string());
            }
            fnames
        }

        /// Reads the default feature table from the file `ipa_all.csv`.
        fn default_feature_table() -> HashMap<String, Vec<i8>> {
            let tab = include_str!("../ipa_all.csv");
            let mut rdr = csv::Reader::from_reader(tab.as_bytes());
            let mut table: HashMap<String, Vec<i8>> = HashMap::new();
            for result in rdr.records() {
                let v = match result {
                    Err(why) => panic!("Bad record: {}", why),
                    Ok(record) => record,
                };
                let k = v[0].to_string();
                let mut tail = Vec::new();
                for i in 1..(v.len()) {
                    tail.push(to_numeric(&v[i]));
                }
                table.insert(k, tail);
            }
            return table;
        }

        /// Reads the feature names from the default file `ipa_all.csv`.
        fn default_feature_names() -> Vec<String> {
            let tab = include_str!("../ipa_all.csv");
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(tab.as_bytes());
            let header = match rdr.records().next() {
                Some(header) => header,
                None => panic!("Empty iterator!"),
            };
            let header = match header {
                Err(why) => panic!("Bad header: {}", why),
                Ok(record) => record,
            };
            let mut fnames = Vec::new();
            for i in 1..(header.len()) {
                let name = header[i].to_string();
                fnames.push(name);
            }
            return fnames;
        }

        /// Returns the feature names associated with a [`FeatureTable`] object.
        pub fn to_fnames(&self) -> Vec<String> {
            return self.fnames.clone();
        }

        /// Parses an IPA `&str` into individual phonemes, based on the feature
        /// table.
        /// 
        /// # Example
        /// 
        /// ```rust
        /// # use rspanphon::featuretable::*;
        /// let ft = FeatureTable::new();
        /// assert_eq!(vec!["pʰ".to_string(), "i".to_string()], ft.phonemes("pʰi"));
        /// ```
        pub fn phonemes(&self, s: &str) -> Vec<String> {
            let ft = &self.ft;
            let mut v: Vec<String> = Vec::new();
            let mut acc: String = String::from("");
            for c in s.chars() {
                acc.push(c);
                if c != '͡' {
                    match ft.get(&acc) {
                        Some(_) => {},
                        None => {
                            match acc.pop() {
                                Some(_) => {
                                    v.push(acc);
                                    acc = c.to_string();
                                },
                                None => {},
                            }
                        }
                    }
                }
            }
            v.push(acc);
            return v;
        }

        /// Takes a vector of phonemes and returns a vector of feature vectors.
        /// 
        /// # Example
        /// 
        /// ```rust
        /// # use rspanphon::featuretable::*;
        /// let ft = FeatureTable::new();
        /// let phonemes = ft.phonemes("kʰul");
        /// assert_eq(3, ft.phonemes_to_vectors(phonemes).len());
        /// ```
        pub fn phonemes_to_vectors(&self, ps: &Vec<String>) -> Vec<Vec<i8>> {
            let mut vs: Vec<Vec<i8>> = Vec::new();
            for p in ps.iter() {
                match &self.ft.get(p) {
                    Some(fv) => vs.push(fv.to_vec()),
                    None => (),
                }
            }
            return vs;
        }

        /// Computes the unweighted feature distance between two vectors. This
        /// runs in worst case O(l * m * n) time (where l is the length of the
        /// feature vectors and m and n are the lengths of the input vectors).
        /// However, the results from the functions that calculate deletion,
        /// insertion, and substitution cost are memoized so that the best case
        /// run time is close to O(m * n).
        pub fn fd(s: &Vec<Vec<i8>>, t: &Vec<Vec<i8>>) -> f64 {
            let n = s.len();
            let m = t.len();
            let k = s[0].len();
            let mut s2: Vec<Vec<i8>> = vec![vec![0; k]];
            let mut t2: Vec<Vec<i8>> = vec![vec![0; k]];
            s2.extend(s.clone());
            t2.extend(t.clone());
            let mut dp = std::iter::repeat(vec![0.0 as f64; m + 1])
                .take(n + 1)
                .collect::<Vec<Vec<f64>>>();
            for i in 1..=n {
                dp[i][0] = dp[i - 1][0] + unweighted_deletion_cost(&s2[i]);
            }
            for j in 1..=m {
                dp[0][j] = dp[0][j - 1] + unweighted_insertion_cost(&t2[j]);
            }
            for i in 1..=n {
                for j in 1..=m {
                    let del = dp[i - 1][j] + unweighted_deletion_cost(&s2[i]);
                    let ins = dp[i][j - 1] + unweighted_insertion_cost(&t2[j]);
                    let sub = dp[i - 1][j - 1]
                        + unweighted_substitution_cost2(&s2[i], &t2[j]);
                    dp[i][j] = sub.min(ins.min(del));
                }
            }
            dp[n][m]
        }

        /// This function wraps [`FeatureTable::fd`], accepting `&str`s directly
        /// instead of vectors of feature vectors. 
        pub fn feature_edit_distance(&self, s1: &str, s2: &str) -> f64 {
            let ps1 = &self.phonemes(s1);
            let ps2 = &self.phonemes(s2);
            let v1 = &self.phonemes_to_vectors(ps1);
            let v2 = &self.phonemes_to_vectors(ps2);
            FeatureTable::fd(v1, v2)
        }

        pub fn to_feature_hashes(&self) -> FeatureHashes {
            let mut fh = FeatureHashes::new();
            fh.fnames = self.fnames.clone();
            for (k, v) in self.ft.clone() {
                let mut f = HashMap::new();
                v.iter().zip(&self.fnames).for_each(|(vl, nm)| {
                    let _ = f.insert(nm.to_string(), *vl);
                });
                fh.fh.insert(k, f);
            }
            return fh;
        }
    }

    #[derive(Debug, Deserialize)]
    pub enum DiaPos {
        #[serde(rename = "pre")]
        Prefix,
        #[serde(rename = "post")]
        Postfix,
    }

    #[derive(Debug, Deserialize)]
    pub struct Combination {
        pub name: String,
        pub combines: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    pub struct DiacriticDefs {
        pub diacritics: Vec<Diacritic>,
        pub combinations: Vec<Combination>,
    }

    #[derive(Debug, Deserialize)]
    pub struct Diacritic {
        pub marker: String,
        pub name: String,
        pub position: DiaPos,
        pub conditions: Vec<HashMap<String, i8>>,
        pub content: HashMap<String, i8>,
    }

    impl Diacritic {
        pub fn affix(&self, base: &str) -> String {
            match &self.position {
                DiaPos::Prefix => format!("{}{}", self.marker, base),
                DiaPos::Postfix => format!("{}{}", base, self.marker),
            }
        }

        pub fn update_ft_map(&self, map: &HashMap<String, i8>) -> HashMap<String, i8> {
            let mut new_map = map.clone();
            for (k, v) in &self.content {
                new_map.insert(k.to_string(), *v);
            }
            new_map
        }

        fn satisfy_conditions(
            conditions: &Vec<HashMap<String, i8>>,
            comparand: &HashMap<String, i8>,
        ) -> bool {
            let result = conditions
                .iter()
                .map(|cond| {
                    cond.iter()
                        .map(|(k, v)| comparand.get(k).unwrap() == v)
                        .all(|x| x)
                })
                .all(|x| x);
            return result;
        }

        pub fn update_seg(
            &self,
            seg: &(String, HashMap<String, i8>),
        ) -> Option<(String, HashMap<String, i8>)> {
            let (ipa, fts) = seg;
            if !(ipa.contains(&self.marker))
                && Diacritic::satisfy_conditions(&self.conditions, &fts)
            {
                let ipa = &self.affix(&ipa);
                let fts = &self.update_ft_map(&fts);
                return Some((ipa.to_string(), fts.clone()));
            } else {
                return None;
            }
        }
    }

    #[derive(Debug)]
    pub struct FeatureHashes {
        pub fh: HashMap<String, HashMap<String, i8>>,
        fnames: Vec<String>,
    }

    impl FeatureHashes {
        pub fn new() -> FeatureHashes {
            FeatureHashes {
                fh: HashMap::new(),
                fnames: Vec::new(),
            }
        }

        pub fn from_base_and_diacritics(base_path: &str, dia_path: &str) -> FeatureHashes {
            let fh = FeatureTable::from_csv(&base_path).to_feature_hashes();
            let dias = FeatureHashes::load_diacritics(&dia_path);
            fh.apply_diacritics(&dias)
        }

        fn map_to_vec(&self, map: &HashMap<String, i8>) -> Vec<i8> {
            self.fnames.iter().map(|name| map[name]).collect()
        }

        pub fn to_feature_table(&self) -> FeatureTable {
            let mut ft = HashMap::new();
            self.fh.iter().for_each(|(k, v)| {
                let _ = ft.insert(k.to_string(), self.map_to_vec(v));
            });
            FeatureTable {
                ft: ft,
                fnames: self.fnames.clone(),
            }
        }

        pub fn apply_diacritics(&self, dias: &Vec<Diacritic>) -> FeatureHashes {
            let mut fh = self.fh.clone();
            let fnames = self.fnames.clone();
            for i in 1..=3 {
                let mut fh_acc: HashMap<String, HashMap<String, i8>> = HashMap::new();
                for (seg, ft_map) in &fh {
                    dias.iter()
                        .filter_map(|d| d.update_seg(&(seg.to_string(), ft_map.clone())))
                        .for_each(|(ipa, map)| {
                            fh_acc.insert(ipa, map).unwrap_or(HashMap::new());
                        });
                }
                if fh_acc.is_empty() {
                    break;
                }
                fh.extend(fh_acc);
                println!("Pass {:?} -- {:?} segments.", i, fh.len());
            }
            FeatureHashes {
                fh: fh,
                fnames: fnames,
            }
        }

        fn load_diacritics(path: &str) -> Vec<Diacritic> {
            let s = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => panic!("Error opening {:?}: {:?}", path, e),
            };
            match serde_yaml::from_str::<DiacriticDefs>(&s) {
                Ok(defs) => return defs.diacritics,
                Err(e) => panic!("Error parsing {:?}: {:?}", path, e),
            }
        }

        fn _unifies(h1: &HashMap<String, i8>, h2: &HashMap<String, i8>) -> bool {
            for (k, v1) in h1 {
                let u = match h2.get(k) {
                    Some(v) => v == v1,
                    _ => true,
                };
                if !u {
                    return false;
                }
            }
            return true;
        }
    }

    #[cached(type="UnboundCache<(Vec<i8>), f64>", create = "{ UnboundCache::new() }")]
    fn unweighted_deletion_cost(v1: &Vec<i8>) -> f64 {
        return v1.iter().map(|&x| x.abs()).sum::<i8>() as f64 / (v1.len() as f64);
    }

    #[cached(type="UnboundCache<(Vec<i8>), f64>", create = "{ UnboundCache::new() }")]
    fn unweighted_insertion_cost(v1: &Vec<i8>) -> f64 {
        return v1.iter().map(|&x| x.abs()).sum::<i8>() as f64 / (v1.len() as f64);
    }

    #[cached(type="UnboundCache<(Vec<i8>, Vec<i8>), f64>", create = "{ UnboundCache::new() }")]
    fn unweighted_substitution_cost2(v1: &Vec<i8>, v2: &Vec<i8>) -> f64 {
        return v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<i8>() as f64
            / (v1.len() as f64);
    }

    #[cached(type="UnboundCache<(Vec<i8>, Vec<i8>), f64>", create = "{ UnboundCache::new() }")]
    fn _unweighted_substitution_cost(v1: &Vec<i8>, v2: &Vec<i8>) -> f64 {
        let d: f64 = v1
            .iter()
            .zip(v2.iter())
            .map(|(x, y)| _feature_diff_ternary(&x, &y))
            .sum::<f64>()
            / (v1.len() as f64);
        return d;
    }

    fn to_numeric(f: &str) -> i8 {
        match f {
            "+" => 1,
            "-" => -1,
            _ => 0,
        }
    }

    fn _feature_diff_ternary(f1: &i8, f2: &i8) -> f64 {
        let d = match (f1, f2) {
            (-1, 1) => 1.0,
            (1, -1) => 1.0,
            _ => 0.0,
        };
        return d;
    }
}

#[cfg(test)]
mod tests {
    use crate::featuretable::*;
    #[test]
    fn ligature_ties() {
        let ft = FeatureTable::new();
        assert_eq!(vec!["t͡s".to_string(), "a".to_string()], ft.phonemes("t͡sa"));
    }
}
