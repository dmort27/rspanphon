use rspanphon::featuretable::FeatureHashes;
use rspanphon::featuretable::FeatureTable;

fn main() {
    let ft = FeatureTable::new();
    // dbg!(&ft);
    // dbg!(&ft.fnames);
    let fh = FeatureHashes::from_base_and_diacritics("ipa_bases.csv", "diacritic_definitions.yml");
    let ft2 = fh.to_feature_table();
    // dbg!(&ft2);
    println! {"fh.fh.len()={:?}", &fh.fh.len()}
    println! {"ft2.ft.len()={:?}", &ft2.ft.len()}
    // println!("{:?}", segs);
    // println!("{:?}", ft.phonemes(&"tone35"));
    // println!(
    //     "dist(pyp, pup)={:?}",
    //     ft.feature_edit_distance(&"pyp", &"pup")
    // );
    // println!(
    //     "dist(pip, pup)={:?}",
    //     ft.feature_edit_distance(&"pip", &"pup")
    // );
    // println!(
    //     "dist(pæp, pup)={:?}",
    //     ft.feature_edit_distance(&"pæp", &"pup")
    // );
    // println!(
    //     "dist(pu, pup)={:?}",
    //     ft.feature_edit_distance(&"pu", &"pup")
    // );
    // println!(
    //     "dist(pup, pupp)={:?}",
    //     ft.feature_edit_distance(&"pup", &"pupp")
    // );
}
