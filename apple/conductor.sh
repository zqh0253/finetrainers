#!/bin/bash

# Changelog
# 0.2.0 - first version (that we track), fixed follow redirects, added CONDUCTOR_INSTALL_DIR, fixed notarycli arch detection, added --help/--version
# 0.2.1 - upgrade notarycli version (cache support, bug-fixes, and compatibility improvements)
# 0.2.2 - upgrade notarycli version (tensorflow-io compatibility fix)
# 0.2.3 - internal change to create multiple notary profiles per internal environment
# 0.2.4 - make script work in Bolt tasks, added -y/--no-prompt option
# 0.2.5 - make script check for presence of aws command for non-ON_BOLT usage
# 0.2.6 - make script check for presence of aws command even for ON_BOLT
# 0.2.7 - Fix bug
# 0.2.8 - upgrade notarycli version (retry improvement fix)
# 0.2.9 - allow users to set ON_BOLT from environement
# 0.3.0 - use homebrew to install notarycli if available
# 0.3.1 - check for internal homebrew build during notarycli install
# 0.3.2 - add `conductor-aodc` binary that is configured for the AODC only endpoint
# 0.3.3 - fixed bug with brew detection, bumped notarycli version
# 0.3.4 - added -f/--force to ensure that we re-install Apple CA/notarycli
# 0.3.5 - fix issue with detecting notarycli/Apple CA (thanks Andy P)
# 0.3.6 - added -d/--doctor to help debug
# 0.3.7 - changed profiles to use absolute path to notarycli
# 0.3.8 - fix notarycli installation through curl, add error trap, cancel uses exit code 3
SCRIPT_VERSION=0.3.8

DEFAULT_TARGET_BIN_DIR=~/.conductor/bin

ASSUME_YES=false
FORCE_INSTALL=false

is_installed() {
    command -v "$1" &> /dev/null
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      printf "\e[1mUsage\e[0m: ./`basename $0`\n"
      echo ""
      echo "This will install Conductor (and notarycli if required) to the default location: $DEFAULT_TARGET_BIN_DIR"
      printf "The default location is already in your \$PATH and will work out of the box, but \e[1mrequires sudo to write\e[0m\n"
      echo ""
      printf "\e[1mUsage\e[0m: CONDUCTOR_INSTALL_DIR=~/your/desired/location ./`basename $0`\n"
      echo ""
      echo "This will install Conductor (and notarycli if required) to your desired location"
      printf "Custom locations will not require sudo to write, but \e[1myou will need to add them to your \$PATH\e[0m (we provide instructions)\n"
      echo "Example: CONDUCTOR_INSTALL_DIR=~/.conductor/bin ./`basename $0`"
      echo ""
      printf "\e[1mOptions\e[0m: \n"
      echo ""
      echo "-f, --force          Force re-install notarycli and the Apple CAs"
      echo "-y, --no-prompt      Assume yes for all dialogs and install without interactive input."
      echo "-v, --version        Print version."
      echo "-d, --doctor         Check the health of your Conductor setup and gather useful system information"
      echo "-h, --help           Print this help message."
      exit 0
      ;;
    -v|--version)
      echo "Conductor Setup Script v${SCRIPT_VERSION}"
      exit 0
      ;;
    -d|--doctor)
      echo "🩺 Conductor Setup Script Doctor"
      echo -e "setup script version:\tv${SCRIPT_VERSION}"
      echo -e "system:\t\t\t$(uname -a)"
      echo -e "term:\t\t\t${TERM:-not set}"
      echo -e "shell:\t\t\t${SHELL:-not set}"
      echo -e "term_program:\t\t${TERM_PROGRAM:-not set}"
      echo -e "bash:\t\t\t$(which bash)"
      if is_installed bash; then
        echo -e "bash version:\t\t$(bash --version | head -1)"
      fi
      echo -e "zsh:\t\t\t$(which zsh)"
      if is_installed zsh; then
        echo -e "zsh version:\t\t$(zsh --version | head -1)"
      fi
      echo -e "aws:\t\t\t$(which aws)"
      if is_installed aws; then
        echo -e "aws version:\t\t$(aws --version)"
      fi
      echo -e "notarycli:\t\t$(which notarycli)"
      if is_installed notarycli; then
        echo -e "notarycli version:\t$(notarycli version)"
      fi
      echo -e "conductor:\t\t$(which conductor)"
      if is_installed brew; then
        echo -e "brew version:\t\t$(brew -v | head -1)"
      fi
      if is_installed aws; then
        echo -e "\nPotentially relevant configured AWS profiles:"
        aws configure list-profiles | grep conductor | paste -s -d ',' -
      fi
      if is_installed grep; then
        echo -e "\nPotentially relevant AWS environment variables set:"
        env | grep -o '^AWS_*[^=]*' | paste -s -d ',' -
      fi
      echo -e "\nHope that helps!"
      exit 0
      ;;
    -y|--no-prompt|--yes)
      ASSUME_YES=true
      shift # past argument
      ;;
    -f|--force)
      FORCE_INSTALL=true
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo "Run with -h/--help to see available options"
      exit 1
      ;;
  esac
done

set -euo pipefail
trap 'catch $? $LINENO' EXIT
catch() {
  if [ "$1" != "0" ]; then
    echo -e "\n💥💀❌ $(tput bold)Oh oh!$(tput sgr0) ❌💀💥"
    echo "It looks like we encountered an error (exit code: $1) on line $2 (v$SCRIPT_VERSION). The setup script did not complete successfully."
  fi
}

BREW_COMMAND="brew"

NOTARYCLI="notarycli"
NOTARY_VERSION="v1.11.0"

CI=${CI:-"false"}

# Allow users to override the target install dir using the environment variable: CONDUCTOR_INSTALL_DIR
TARGET_BIN_DIR=${CONDUCTOR_INSTALL_DIR:-~/.conductor/bin/}

CONDUCTOR_DIR=~/.conductor
APPLE_CA=$CONDUCTOR_DIR/apple_ca.crt
AWS_DIR=~/.aws
AWS_CONFIG=$AWS_DIR/config
AWS_CREDENTIALS=$AWS_DIR/credentials

# If the user provides a different directory, assume its because they don't want sudo
if [[ "${TARGET_BIN_DIR}" != "${DEFAULT_TARGET_BIN_DIR}" ]]; then
  mkdir -p "$TARGET_BIN_DIR"
  TARGET_BIN_DIR=$(realpath "$TARGET_BIN_DIR")
fi

## For internal use by the Conductor and infra teams. This will install non-user accessible clusters.
INSTALL_CONDUCTOR_INTERNAL_ENVIRONMENTS=false

if [[ "$ASSUME_YES" == "true" ]]; then
  echo "ℹ️ Running with --no-prompt"
fi

if [[ "$FORCE_INSTALL" == "true" ]]; then
  echo "ℹ️ Running with --force install"
fi

if [[ "$INSTALL_CONDUCTOR_INTERNAL_ENVIRONMENTS" == "true" ]]; then
  echo "ℹ️ Will also install all internal environments"
fi


echo -e "
  ▄▄█▀▀▀█▄█                        ▀███                     ██
▄██▀     ▀█                          ██                     ██
██▀       ▀ ▄██▀██▄▀████████▄   ▄█▀▀███ ▀███  ▀███  ▄██▀████████  ▄██▀██▄▀███▄███
██         ██▀   ▀██ ██    ██ ▄██    ██   ██    ██ ██▀  ██  ██   ██▀   ▀██ ██▀ ▀▀
██▄        ██     ██ ██    ██ ███    ██   ██    ██ ██       ██   ██     ██ ██
▀██▄     ▄▀██▄   ▄██ ██    ██ ▀██    ██   ██    ██ ██▄    ▄ ██   ██▄   ▄██ ██
  ▀▀█████▀  ▀█████▀▄████  ████▄▀████▀███▄ ▀████▀███▄█████▀  ▀████ ▀█████▀▄████▄
"

ON_BOLT=${ON_BOLT:-false}
# Bolt will always populate `$TASK_ID`, but since its name is a tad too
# generic, let's be safe and ensure there is also a `$BOLT_ARTIFACT_DIR` set.
# Additionally, we make this script work without interactivity on Bolt.
if [ "${TASK_ID:-notset}" != "notset" ] && [ "${BOLT_ARTIFACT_DIR:-notset}" != "notset" ]; then
  ON_BOLT=true
  ASSUME_YES=true
fi

if [ "$EUID" -eq 0 ]; then
  # Bolt tasks run always run as the root user, so skip the check there
  if [ "$ON_BOLT" = true ]; then
    echo -e "
            ╔══╗     ╔╗  ╔╗
            ║╔╗║     ║║ ╔╝╚╗
╔══╗╔═╗     ║╚╝╚╗╔══╗║║ ╚╗╔╝
║╔╗║║╔╗╗    ║╔═╗║║╔╗║║║  ║║
║╚╝║║║║║    ║╚═╝║║╚╝║║╚╗ ║╚╗
╚══╝╚╝╚╝    ╚═══╝╚══╝╚═╝ ╚═╝
"
  elif [[ "$CI" == "true" ]]; then
    echo "Detected CI: skipping root check";
  else
    echo "⛔ Please do not run the script as root."
    exit 1
  fi
fi

echo "setup script version: $SCRIPT_VERSION"
echo "We are going to first verify (and setup) the pre-requisites"

if [[ -t 1 ]]
then
    tty_escape() { printf "\033[%sm" "$1"; }
else
    tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_red="$(tty_mkbold 31)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

if ! is_installed aws; then
  echo -e "❌ ${tty_red}AWS Cli.${tty_reset}\nPlease install by following the instructions here: ${tty_bold}https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html${tty_reset}"
  exit 1
else
  echo "✅ AWS CLI found"
fi

# make sure the Apple CAs are available
if [[ $FORCE_INSTALL == "true" ]] || [[ ! -f "$APPLE_CA" ]]; then
  mkdir -p $CONDUCTOR_DIR
  touch $APPLE_CA
  echo "✔️ Installing Apple CAs"
  # source: https://github.pie.apple.com/crypto-services/trust-apple-corp-root-cas
  cat > $APPLE_CA <<- END
-----BEGIN CERTIFICATE-----
MIIDsTCCApmgAwIBAgIIFJlrSmrkQKAwDQYJKoZIhvcNAQELBQAwZjEgMB4GA1UE
AwwXQXBwbGUgQ29ycG9yYXRlIFJvb3QgQ0ExIDAeBgNVBAsMF0NlcnRpZmljYXRp
b24gQXV0aG9yaXR5MRMwEQYDVQQKDApBcHBsZSBJbmMuMQswCQYDVQQGEwJVUzAe
Fw0xMzA3MTYxOTIwNDVaFw0yOTA3MTcxOTIwNDVaMGYxIDAeBgNVBAMMF0FwcGxl
IENvcnBvcmF0ZSBSb290IENBMSAwHgYDVQQLDBdDZXJ0aWZpY2F0aW9uIEF1dGhv
cml0eTETMBEGA1UECgwKQXBwbGUgSW5jLjELMAkGA1UEBhMCVVMwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQC1O+Ofah0ORlEe0LUXawZLkq84ECWh7h5O
7xngc7U3M3IhIctiSj2paNgHtOuNCtswMyEvb9P3Xc4gCgTb/791CEI/PtjI76T4
VnsTZGvzojgQ+u6dg5Md++8TbDhJ3etxppJYBN4BQSuZXr0kP2moRPKqAXi5OAYQ
dzb48qM+2V/q9Ytqpl/mUdCbUKAe9YWeSVBKYXjaKaczcouD7nuneU6OAm+dJZcm
hgyCxYwWfklh/f8aoA0o4Wj1roVy86vgdHXMV2Q8LFUFyY2qs+zIYogVKsRZYDfB
7WvO6cqvsKVFuv8WMqqShtm5oRN1lZuXXC21EspraznWm0s0R6s1AgMBAAGjYzBh
MB0GA1UdDgQWBBQ1ICbOhb5JJiAB3cju/z1oyNDf9TAPBgNVHRMBAf8EBTADAQH/
MB8GA1UdIwQYMBaAFDUgJs6FvkkmIAHdyO7/PWjI0N/1MA4GA1UdDwEB/wQEAwIB
BjANBgkqhkiG9w0BAQsFAAOCAQEAcwJKpncCp+HLUpediRGgj7zzjxQBKfOlRRcG
+ATybdXDd7gAwgoaCTI2NmnBKvBEN7x+XxX3CJwZJx1wT9wXlDy7JLTm/HGa1M8s
Errwto94maqMF36UDGo3WzWRUvpkozM0mTcAPLRObmPtwx03W0W034LN/qqSZMgv
1i0use1qBPHCSI1LtIQ5ozFN9mO0w26hpS/SHrDGDNEEOjG8h0n4JgvTDAgpu59N
CPCcEdOlLI2YsRuxV9Nprp4t1WQ4WMmyhASrEB3Kaymlq8z+u3T0NQOPZSoLu8cX
akk0gzCSjdeuldDXI6fjKQmhsTTDlUnDpPE2AAnTpAmt8lyXsg==
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIICRTCCAcugAwIBAgIIE0aVDhdcN/0wCgYIKoZIzj0EAwMwaDEiMCAGA1UEAwwZ
QXBwbGUgQ29ycG9yYXRlIFJvb3QgQ0EgMjEgMB4GA1UECwwXQ2VydGlmaWNhdGlv
biBBdXRob3JpdHkxEzARBgNVBAoMCkFwcGxlIEluYy4xCzAJBgNVBAYTAlVTMB4X
DTE2MDgxNzAxMjgwMVoXDTM2MDgxNDAxMjgwMVowaDEiMCAGA1UEAwwZQXBwbGUg
Q29ycG9yYXRlIFJvb3QgQ0EgMjEgMB4GA1UECwwXQ2VydGlmaWNhdGlvbiBBdXRo
b3JpdHkxEzARBgNVBAoMCkFwcGxlIEluYy4xCzAJBgNVBAYTAlVTMHYwEAYHKoZI
zj0CAQYFK4EEACIDYgAE6ROVmqXFAFCLpuLD3loNJwfuxX++VMPgK5QmsUuMmjGE
/3NWOUGitN7kNqfq62ebPFUqC1jUZ3QzyDt3i104cP5Z5jTC6Js4ZQxquyzTNZiO
emYPrMuIRYHBBG8hFGQxo0IwQDAdBgNVHQ4EFgQU1u/BzWSVD2tJ2l3nRQrweevi
XV8wDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAQYwCgYIKoZIzj0EAwMD
aAAwZQIxAKJCrFQynH90VBbOcS8KvF1MFX5SaMIVJtFxmcJIYQkPacZIXSwdHAff
i3+/qT+DhgIwSoUnYDwzNc4iHL30kyRzAeVK1zOUhH/cuUAw/AbOV8KDNULKW1Nc
xW6AdqJp2u2a
-----END CERTIFICATE-----
END
else
  echo "✅ Apple CAs found"
fi

NOTARYCLI_FULLPATH=""
if is_installed "$NOTARYCLI"; then
  NOTARYCLI_FULLPATH=$(which "$NOTARYCLI")
fi

# make sure notarycli is installed
if [[ $FORCE_INSTALL == "true" ]] || ! is_installed "$NOTARYCLI"; then
    echo "ℹ️ Going to install Notary CLI"
    if [ "$ASSUME_YES" = false ]; then
      read -rp "Would you like us to install notarycli for you? (y/n): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 3
    fi

    install_with_curl=true
    if [[ $(command -v $BREW_COMMAND) != "" ]]; then
      if [[ $($BREW_COMMAND -v | head -1) == *-apple* ]]; then
        $BREW_COMMAND install apple/turi/notarycli
        NOTARYCLI_FULLPATH="$($BREW_COMMAND --prefix)/bin/notarycli"
        install_with_curl=false
      else
        echo "⚠️ Non-Apple Homebrew detected. Please migrate to internal fork: https://github.pie.apple.com/homebrew/brew."
        if [ "$ASSUME_YES" = false ]; then
          read -rp "Would you like to proceed using curl instead? (y/n): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 3
        fi
      fi
    fi

    if [ "$install_with_curl" = true ]; then
      binary=""
      if [ "$(uname)" == "Darwin" ]; then
        if [[ $(uname -p) == 'arm' ]]; then
          binary="notarycli-darwin-arm64"
        else
          binary="notarycli-darwin-amd64"
        fi
      elif [ "$(expr substr "$(uname -s)" 1 5)" == "Linux" ]; then
        if [[ $(uname -p) == 'arm' ]]; then
          binary="notarycli-linux-arm64"
        else
          binary="notarycli-linux-amd64"
        fi
      else
        >&2 echo "Unsupported platform! Only supports Darwin or Linux"
        exit 1
      fi
      curl -Ls "https://artifacts.apple.com/polymer-generic-local/notary/notarycli/${NOTARY_VERSION}/${binary}.tar.gz" -o "/tmp/${binary}.tar.gz" && \
        tar -xzvf "/tmp/${binary}.tar.gz" -C /tmp/ &>/dev/null && \
        chmod +x "/tmp/${binary}" && \
        mv "/tmp/${binary}" "${TARGET_BIN_DIR}"/notarycli
      NOTARYCLI_FULLPATH="${TARGET_BIN_DIR}"/notarycli
    fi
    echo "✔️ notarycli installed"
else
  echo "✅ notarycli found"
fi

if [[ $NOTARYCLI_FULLPATH == "" ]]; then
  echo "Looks like we couldn't install/find your 'notarycli', failing"
  exit 127
fi

if [ ! -f $AWS_CONFIG ]; then
  mkdir -p ${AWS_DIR}
  if [ "$ASSUME_YES" = false ]; then
    read -rp "It looks like you don't have an ${AWS_CONFIG}. Would you like us to create it for you? (y/n): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 3
  fi
  touch ${AWS_CONFIG}
fi

if [ ! -f $AWS_CREDENTIALS ]; then
  mkdir -p ${AWS_DIR}
  if [ "$ASSUME_YES" = false ]; then
    read -rp "It looks like you don't have an ${AWS_CREDENTIALS}. Would you like us to create it for you? (y/n): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 3
  fi
  touch ${AWS_CREDENTIALS}
fi

set +e # grep returns errors for no matches
begin_marker=$(grep -n BEGIN-SETUP-CONDUCTOR ${AWS_CONFIG} | cut -f1 -d:)
end_marker=$(grep -n END-SETUP-CONDUCTOR ${AWS_CONFIG} | cut -f1 -d:)
has_old_profile=$(grep -n conductor-notary ${AWS_CONFIG} | cut -f1 -d:)
set -e

if [[ ${has_old_profile} != "" && ${begin_marker} == "" && ${end_marker} == "" ]]; then
  echo "⚠️ You have an old 'conductor-notary' profile. Please remove it by manually editing ${AWS_CONFIG} and removing the '[profile conductor-notary]'. This is a one time requirement."
  exit 1
fi

if [[ "${begin_marker}" == "" && "${end_marker}" == "" ]]; then
  : # No old profile. Nothing to do.
elif [[ "${begin_marker}" != "" && "${end_marker}" != "" ]]; then
  sed -i.bak "${begin_marker},${end_marker}d" $AWS_CONFIG
elif [[ "${begin_marker}" == "" ]]; then
  echo "You're missing a BEGIN-SETUP-CONDUCTOR marker in your ${AWS_CONFIG}. Add it and try again."
  exit 1
elif [[ "${end_marker}" == "" ]]; then
  echo "You're missing a END-SETUP-CONDUCTOR marker in your ${AWS_CONFIG}. Add it and try again."
  exit 1
fi

# configure the notary credential process profile
echo "✔️ Installing AWS conductor-notary profile"
if [ "$INSTALL_CONDUCTOR_INTERNAL_ENVIRONMENTS" = true ]; then
cat >> $AWS_CONFIG <<- END

# BEGIN-SETUP-CONDUCTOR
# Added automatically via the setup-conductor.sh script.
# Any changes in this section will be replaced if you run the script again.
# If you want to manually manage this section remove the BEGIN and END markers.

[profile conductor-notary]
ca_bundle = ${APPLE_CA}
credential_process = ${NOTARYCLI_FULLPATH} issue -o conductor --audience=aprn:apple:turi::notary:application:conductor
region = conductor

[profile conductor-notary-dev]
ca_bundle = ${APPLE_CA}
credential_process = ${NOTARYCLI_FULLPATH} issue -o conductor --audience=aprn:apple:turi::notary:application:conductor --conductor-endpoint https://conductor-dev.data.apple.com
region = conductor

[profile conductor-notary-infra]
ca_bundle = ${APPLE_CA}
credential_process = ${NOTARYCLI_FULLPATH} issue -o conductor --audience=aprn:apple:turi::notary:application:conductor --conductor-endpoint https://conductor-infra.data.apple.com
region = conductor

[profile conductor-notary-stg]
ca_bundle = ${APPLE_CA}
credential_process = ${NOTARYCLI_FULLPATH} issue -o conductor --audience=aprn:apple:turi::notary:application:conductor --conductor-endpoint https://conductor-stg.data.apple.com
region = conductor

# END-SETUP-CONDUCTOR
END
else
cat >> $AWS_CONFIG <<- END

# BEGIN-SETUP-CONDUCTOR
# Added automatically via the setup-conductor.sh script.
# Any changes in this section will be replaced if you run the script again.
# If you want to manually manage this section remove the BEGIN and END markers.

[profile conductor-notary]
ca_bundle = ${APPLE_CA}
credential_process = ${NOTARYCLI_FULLPATH} issue -o conductor --audience=aprn:apple:turi::notary:application:conductor
region = conductor

# END-SETUP-CONDUCTOR
END

set +e # grep returns errors for no matches
begin_marker=$(grep -n BEGIN-SETUP-CONDUCTOR ${AWS_CREDENTIALS} | cut -f1 -d:)
end_marker=$(grep -n END-SETUP-CONDUCTOR ${AWS_CREDENTIALS} | cut -f1 -d:)
has_old_profile=$(grep -n conductor-sidecar ${AWS_CREDENTIALS} | cut -f1 -d:)
set -e

if [[ ${has_old_profile} != "" && ${begin_marker} == "" && ${end_marker} == "" ]]; then
  echo "⚠️ You have old 'conductor-sidecar' credentials. Please remove it by manually editing ${AWS_CREDENTIALS} and removing the '[conductor-sidecar]'. This is a one time requirement."
  exit 1
fi

if [[ "${begin_marker}" == "" && "${end_marker}" == "" ]]; then
  : # No old profile. Nothing to do.
elif [[ "${begin_marker}" != "" && "${end_marker}" != "" ]]; then
  sed -i.bak "${begin_marker},${end_marker}d" $AWS_CREDENTIALS
elif [[ "${begin_marker}" == "" ]]; then
  echo "You're missing a BEGIN-SETUP-CONDUCTOR marker in your ${AWS_CREDENTIALS}. Add it and try again."
  exit 1
elif [[ "${end_marker}" == "" ]]; then
  echo "You're missing a END-SETUP-CONDUCTOR marker in your ${AWS_CREDENTIALS}. Add it and try again."
  exit 1
fi

# configure the conductor-sidecar credentials
echo "✔️ Installing AWS conductor-sidecar credentials"
cat >> $AWS_CREDENTIALS <<- END
# BEGIN-SETUP-CONDUCTOR
# Added automatically via the setup-conductor.sh script.
# Any changes in this section will be replaced if you run the script again.
# If you want to manually manage this section remove the BEGIN and END markers.

[conductor-sidecar]
aws_access_key_id=conductor-access
aws_secret_access_key=conductor-secret

# END-SETUP-CONDUCTOR
END

cat > $TARGET_BIN_DIR/conductor-sidecar <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true exec aws --profile conductor-sidecar --endpoint-url http://localhost:8080 "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor-sidecar
echo "✔️ Installed conductor-sidecar. Use the 'conductor-sidecar' command if you want to access a locally running conductor-as-a-sidecar image."

cat > $TARGET_BIN_DIR/conductor-aodc <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true AWS_CA_BUNDLE=${APPLE_CA} exec aws --profile conductor-notary --endpoint-url https://aodc.conductor.data.apple.com "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor-aodc
echo "✔️ Installed conductor-aodc. Use the 'conductor-aodc' command like you would 'aws' to access the production cluster through (only routes to AODC servers). See https://at.apple.com/conductor-aodc for more info."

cat > $TARGET_BIN_DIR/conductor <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true AWS_CA_BUNDLE=${APPLE_CA} exec aws --profile conductor-notary --endpoint-url https://conductor.data.apple.com "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor
echo "✔️ Installed conductor. Use the 'conductor' command like you would 'aws' to access our production cluster"

cat > $TARGET_BIN_DIR/conductor-qa <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true AWS_CA_BUNDLE=${APPLE_CA} exec aws --profile conductor-notary --endpoint-url https://conductor-qa.data.apple.com "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor-qa
echo "✔️ Installed conductor-qa. Use the 'conductor-qa' command like you would 'aws' to access our QA cluster"

if [ "$INSTALL_CONDUCTOR_INTERNAL_ENVIRONMENTS" = true ]; then

cat > $TARGET_BIN_DIR/conductor-dev <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true AWS_CA_BUNDLE=${APPLE_CA} exec aws --profile conductor-notary-dev --endpoint-url https://conductor-dev.data.apple.com "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor-dev
echo "✔️ Installed conductor-dev"

cat > $TARGET_BIN_DIR/conductor-infra <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true AWS_CA_BUNDLE=${APPLE_CA} exec aws --profile conductor-notary-infra --endpoint-url https://conductor-infra.data.apple.com "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor-infra
echo "✔️ Installed conductor-infra"

cat > $TARGET_BIN_DIR/conductor-staging <<- END
#!/bin/bash
# added automatically via the setup-conductor.sh script
set -eu
AWS_EC2_METADATA_DISABLED=true AWS_CA_BUNDLE=${APPLE_CA} exec aws --profile conductor-notary-stg --endpoint-url https://conductor-stg.data.apple.com "\$@"
END
chmod +x $TARGET_BIN_DIR/conductor-staging
echo "✔️ Installed conductor-staging"

fi

if [[ "${TARGET_BIN_DIR}" != "${DEFAULT_TARGET_BIN_DIR}" ]]; then
  RCFILE=""
  if [[ ${SHELL} == "/bin/zsh" ]]; then
    RCFILE="~/.zshrc"
  elif [[ ${SHELL} == "/bin/bash" ]]; then
    RCFILE="~/.bashrc"
  else
    RCFILE="shell initialization/config file (e.g., ~/.zshrc or ~/.bashrc)"
  fi

  echo -e "\n"
  echo "⚠️ REMINDER ⚠️"
  echo "You installed all binaries to custom location: ${TARGET_BIN_DIR}"
  echo "You will probably want to add it to your \$PATH "
  echo "For example, by adding the following line to your ${RCFILE}"
  echo "export PATH='${TARGET_BIN_DIR}:\$PATH"
else
  # Bolt is already running as root (and there is no sudo anyways)
  if [ "$EUID" -eq 0 ]; then
    ln -sf "${TARGET_BIN_DIR}"/* /usr/local/bin/
  else
    echo "ℹ️ We will now need sudo to link all the binaries to a place on your \$PATH (/usr/local/bin)"
    sudo ln -sf "${TARGET_BIN_DIR}"/* /usr/local/bin/
  fi
fi

echo "🚀 Installed all binaries!"

fi
