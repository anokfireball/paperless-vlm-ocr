// Keeps the plugin in step with the Paperless-ngx release Renovate bumps.
//
// Renovate runs this as a post-upgrade task of the
// `ghcr.io/paperless-ngx/paperless-ngx` dependency, from the repository root:
//   node .github/scripts/renovate-paperless-bump.mjs
//
// It reads VERIFIED_PAPERLESS from site/paperless_vlm_ocr/__init__.py. A patch
// update of an already supported Paperless minor changes nothing: CI re-verifies
// the plugin against the new image. A new minor is added to the
// SUPPORTED_PAPERLESS set literal (sorted, older minors kept) and the plugin's
// patch version is bumped in all three places - __version__, the dist-info
// directory name and METADATA's Version line - so the merge publishes a new
// image. Missing anchors exit 1 without writing; rerunning after a successful
// bump is a no-op.
import { existsSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const SITE = 'site';
const PACKAGE = 'paperless_vlm_ocr';
const INIT_FILE = `${SITE}/${PACKAGE}/__init__.py`;

function main() {
  const initBody = readFileSync(root + INIT_FILE, 'utf8');

  const verified = /^VERIFIED_PAPERLESS = "(\d+)\.(\d+)\.(\d+)"$/m.exec(initBody);
  if (!verified) fail(`no VERIFIED_PAPERLESS = "X.Y.Z" line in ${INIT_FILE}`);
  const [verifiedLine, verifiedMajor, verifiedMinor, verifiedPatch] = verified;
  const major = Number(verifiedMajor);
  const minor = Number(verifiedMinor);

  const supported = /^SUPPORTED_PAPERLESS = \{([^}]*)\}$/m.exec(initBody);
  if (!supported) fail(`no SUPPORTED_PAPERLESS = {...} set literal in ${INIT_FILE}`);
  const tuples = [...supported[1].matchAll(/\(\s*(\d+)\s*,\s*(\d+)\s*\)/g)].map(
    (match) => [Number(match[1]), Number(match[2])],
  );
  if (tuples.length === 0) fail(`SUPPORTED_PAPERLESS in ${INIT_FILE} lists no versions`);
  const leftover = supported[1]
    .replace(/\(\s*\d+\s*,\s*\d+\s*\)/g, '')
    .replace(/[\s,]/g, '');
  if (leftover) fail(`cannot parse SUPPORTED_PAPERLESS in ${INIT_FILE}: found "${leftover}"`);

  if (tuples.some(([knownMajor, knownMinor]) => knownMajor === major && knownMinor === minor)) {
    console.log(
      `renovate-paperless-bump: verified Paperless ${major}.${minor}.${verifiedPatch}; ` +
        `minor ${major}.${minor} is already in SUPPORTED_PAPERLESS; nothing to change.`,
    );
    return;
  }

  const version = /^__version__ = "(\d+)\.(\d+)\.(\d+)"$/m.exec(initBody);
  if (!version) fail(`no __version__ = "X.Y.Z" line in ${INIT_FILE}`);
  const [versionLine, pluginMajor, pluginMinor, pluginPatch] = version;
  const oldVersion = `${pluginMajor}.${pluginMinor}.${Number(pluginPatch)}`;
  const newVersion = `${pluginMajor}.${pluginMinor}.${Number(pluginPatch) + 1}`;

  const distDirs = readdirSync(root + SITE, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.isDirectory() &&
        entry.name.startsWith(`${PACKAGE}-`) &&
        entry.name.endsWith('.dist-info'),
    )
    .map((entry) => entry.name);
  const oldDist = `${PACKAGE}-${oldVersion}.dist-info`;
  const newDist = `${PACKAGE}-${newVersion}.dist-info`;
  if (distDirs.length !== 1 || distDirs[0] !== oldDist) {
    fail(
      `expected only ${SITE}/${oldDist}, found ${distDirs.join(', ') || 'no dist-info directory'}`,
    );
  }

  const metadataFile = `${SITE}/${oldDist}/METADATA`;
  if (!existsSync(root + metadataFile)) fail(`${metadataFile} is missing`);
  const metadata = readFileSync(root + metadataFile, 'utf8');
  const metadataVersion = /^Version: .*$/m.exec(metadata);
  if (!metadataVersion) fail(`no Version: line in ${metadataFile}`);
  if (metadataVersion[0] !== `Version: ${oldVersion}`) {
    fail(`${metadataFile} has "${metadataVersion[0]}", expected "Version: ${oldVersion}"`);
  }

  const nextSet = [...tuples, [major, minor]]
    .sort(([aMajor, aMinor], [bMajor, bMinor]) => aMajor - bMajor || aMinor - bMinor)
    .map(([setMajor, setMinor]) => `(${setMajor}, ${setMinor})`)
    .join(', ');

  writeFileSync(
    root + INIT_FILE,
    initBody
      .replace(versionLine, `__version__ = "${newVersion}"`)
      .replace(supported[0], `SUPPORTED_PAPERLESS = {${nextSet}}`),
  );
  writeFileSync(
    root + metadataFile,
    metadata.replace(metadataVersion[0], `Version: ${newVersion}`),
  );
  renameSync(root + `${SITE}/${oldDist}`, root + `${SITE}/${newDist}`);

  console.log(
    `renovate-paperless-bump: verified Paperless ${major}.${minor}.${verifiedPatch}; ` +
      `added (${major}, ${minor}) to SUPPORTED_PAPERLESS.`,
  );
  console.log(
    `renovate-paperless-bump: plugin ${oldVersion} -> ${newVersion}; ` +
      `renamed ${SITE}/${oldDist} -> ${SITE}/${newDist}.`,
  );
}

function fail(message) {
  console.error(`renovate-paperless-bump: ${message}`);
  process.exit(1);
}

main();
