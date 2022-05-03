class Spectrum:
	
	def __init__(self, header, spec, rms=None):
		self.beam = Beam.from_fits_header(header)
		self.coord = WCS(header).spectral
		self.restfreq = header['RESTFRQ'] * u.Hz
		self.spectrum = spec
		self.rms = rms