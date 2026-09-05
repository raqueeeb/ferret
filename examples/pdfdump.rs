fn main() {
    let path = std::env::args().nth(1).expect("usage: pdfdump FILE.pdf");
    let bytes = std::fs::read(&path).unwrap();
    let text = ferret::extract::pdf::extract(&bytes);
    let toks: Vec<String> = {
        let mut v = Vec::new();
        ferret::tokenize::tokenize(&text, |t, _| v.push(t.to_string()));
        v
    };
    eprintln!("ferret tokens: {}", toks.len());
    println!("{}", toks.join(" "));
}
