import math


class AverageMeter:
    """Track a running average weighted by the number of scored samples."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


class BaseMetric:
    def __init__(self):
        self.meter = AverageMeter()

    def reset(self):
        self.meter.reset()
        
    def calculate_score(self, batch, update=True):
        """
        Batch: {"gt_im": [PIL Image], "gen_im": [Image]}
        """
        from tqdm import tqdm

        values = []
        batch_size = len(next(iter(batch.values())))
        for index in tqdm(range(batch_size)):
            kwargs = {}
            for key in ["gt_im", "gen_im", "gt_svg", "gen_svg", "caption"]:
                if key in batch:
                    kwargs[key] = batch[key][index]
            try:
                measure = self.metric(**kwargs)
            except Exception as e:
                print("Error calculating metric: {}".format(e))
                continue
            if math.isnan(measure):
                continue
            values.append(measure)

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan")

        score = sum(values) / len(values)
        if update:
            self.meter.update(score, len(values))
            return self.meter.avg, values
        else:
            return score, values

    def metric(self, **kwargs):
        """
        This method should be overridden by subclasses to provide the specific metric computation.
        """
        raise NotImplementedError("The metric method must be implemented by subclasses.")
    
    def get_average_score(self):
        return self.meter.avg
