import torch
from torch._C import Value
from torch.autograd import Function
# from ..box_utils import decode, nms
from ..box_utils import decode, nms
# from data import voc as cfg
from ...voc import voc as cfg


class Detect(Function):
    # RuntimeError: legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with
    # static forward method. (Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)

    # def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
    #     self.num_classes = num_classes 
    #     self.background_label = bkg_label
    #     self.top_k = top_k
    #     # Parameters used in nms.
    #     self.nms_thresh = nms_thresh
    #     if nms_thresh <= 0:
    #         raise ValueError('nms_threshold must be non negative.')
    #     self.conf_thresh = conf_thresh
    #     self.variance = cfg['variance']
    
    @staticmethod
    def forward(self,num_classes, bkg_label, top_k, conf_thresh, nms_thresh, loc_data, conf_data, prior_data):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k 
        # parameters used in nms 
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        # pytorch 1.5.0 support new-style function 
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


#   """At test time, Detect is the final layer of SSD.  Decode location preds,
#     apply non-maximum suppression to location predictions based on conf
#     scores and threshold to a top_k number of output predictions for both
#     confidence score and locations.
#     """