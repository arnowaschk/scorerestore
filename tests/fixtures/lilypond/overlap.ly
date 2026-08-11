\version "2.26.0"

\header {
  title = "ScoreRestore Mask Fixture"
  composer = "Public-domain style test material"
  tagline = ##f
}

\paper {
  paper-width = 120\mm
  paper-height = 80\mm
  top-margin = 6\mm
  bottom-margin = 6\mm
  left-margin = 8\mm
  right-margin = 8\mm
  indent = 0\mm
}

fixtureMelody = \relative c' {
  \key c \major
  \time 4/4
  c4( d) e-> f |
  g1\fermata
}

fixtureLyrics = \lyricmode {
  Mask text stays aligned.
}

\score {
  <<
    \new Staff \new Voice = "fixtureVoice" { \fixtureMelody }
    \new Lyrics \lyricsto "fixtureVoice" { \fixtureLyrics }
  >>
  \layout { }
}

