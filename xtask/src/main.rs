use std::process::Command;

fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "help".to_owned());

    let result = match command.as_str() {
        "dev" => run_dev(),
        _ => {
            eprintln!("usage: cargo dev");
            Ok(())
        }
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_dev() -> Result<(), String> {
    let host_target = host_target()?;

    run("cargo", &["+stable", "fmt", "--all"])?;
    run(
        "cargo",
        &["+stable", "test", "--lib", "--target", &host_target],
    )?;
    run("cargo", &["build", "--target", "xtensa-esp32s3-none-elf"])?;
    run(
        "cargo",
        &[
            "clippy",
            "--target",
            "xtensa-esp32s3-none-elf",
            "--bin",
            "SmartGlove",
            "--",
            "-D",
            "warnings",
        ],
    )?;

    Ok(())
}

fn host_target() -> Result<String, String> {
    let output = Command::new("rustc")
        .args(["-vV"])
        .output()
        .map_err(|error| format!("failed to invoke rustc: {error}"))?;

    if !output.status.success() {
        return Err("failed to query rustc host target".to_owned());
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("rustc output was not valid utf-8: {error}"))?;

    stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(ToOwned::to_owned))
        .ok_or_else(|| "could not determine rustc host target".to_owned())
}

fn run(program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .args(args)
        .status()
        .map_err(|error| format!("failed to run `{program} {}`: {error}", args.join(" ")))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "`{program} {}` failed with status {status}",
            args.join(" ")
        ))
    }
}
