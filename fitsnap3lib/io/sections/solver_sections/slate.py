from fitsnap3lib.io.sections.sections import Section


class Slate(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['method', 'alpha', 'alpha_curvature',
            'max_iter', 'rtol', 'atol',
            'alphabig', 'alphasmall', 'lambdabig', 'lambdasmall',
            'directmethod', 'scap', 'scai', 'logcut', 'threshold_lambda']
                       
        self._check_section()
        self._check_if_used("SOLVER", "solver", "SLATE")
        
        # Method selection: RIDGE (default) or ARD
        self.method = self.get_value("SLATE", "method", "RIDGE", "str")

        # alpha for RIDGE
        self.alpha = self.get_value("SLATE", "alpha", "1e-8", "float")
        self.alpha_curvature = self.get_value("SLATE", "alpha_curvature", "1e-8", "float")

        # ARD parameters - matching legacy ARD section
        # Maximum number of iterations
        self.max_iter = self.get_value("SLATE", "max_iter", "10", "int")
        
        # Stop the algorithm if w has converged
        self.rtol = self.get_value("SLATE", "rtol", "1e-3", "float")
        self.atol = self.get_value("SLATE", "atol", "1e-6", "float")
        
        # Direct method hyperparameters (used if directmethod=1)
        self.alphabig = self.get_value("SLATE", "alphabig", "1.0E-12", "float")
        self.alphasmall = self.get_value("SLATE", "alphasmall", "1.0E-14", "float")
        self.lambdabig = self.get_value("SLATE", "lambdabig", "1.0E-6", "float")
        self.lambdasmall = self.get_value("SLATE", "lambdasmall", "1.0E-6", "float")
                
        # Lambda threshold for removing (pruning) weights with high precision from the computation.
        # If not specified, will be auto-computed as 10^(int(abs(log10(ap))) + logcut)
        self.threshold_lambda = self.get_value("SLATE", "threshold_lambda", "0", "float")
        
        # Adaptive hyperparameter mode (0=adaptive using scap/scai, 1=direct using alphabig/lambdasmall)
        self.directmethod = self.get_value("SLATE", "directmethod", "0", "int")
        
        # Scaling factors for adaptive hyperparameters (used if directmethod=0)
        self.scap = self.get_value("SLATE", "scap", "1e-3", "float")
        self.scai = self.get_value("SLATE", "scai", "1e-3", "float")
        
        # Log cutoff for auto-computing threshold_lambda (used if threshold_lambda not specified)
        self.logcut = self.get_value("SLATE", "logcut", "0.3", "float")
        
        self.delete()
