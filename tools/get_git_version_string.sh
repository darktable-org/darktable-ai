#!/bin/sh
#
# Compute a darktable-style version string from the repo's git history.
# Mirrors darktable/tools/get_git_version_string.sh so model releases
# carry the same version shape as the host app.
#
# Examples of output:
#   release-5.6.0 tagged commit      -> "5.6.0"
#   47 commits past release-5.6.0    -> "5.6.0+47~gXXXXXXX"
#   dirty working tree               -> "5.6.0+47~gXXXXXXX~dirty"
#   no release-* tag at all          -> bare commit hash from git describe
#   not a git repo / unknown         -> "unknown-version" (exits 0)
#
# Used by CI workflows for the nightly version label and by maintainer
# scripts that need to print "what version am I on".

VERSION="$(git describe --tags --dirty --match release-* 2>/dev/null)"

if [ $? -eq 0 ] ;
then
  echo "$VERSION" | sed 's,^release-,,;s,-,+,;s,-,~,;'
  exit 0
fi

# shallow clones may have no tags; fall back to the bare commit hash
VERSION="$(git describe --always --dirty 2>/dev/null)"
if [ $? -eq 0 ] ;
then
  echo "$VERSION"
  exit 0
fi

echo "unknown-version"
exit 0
