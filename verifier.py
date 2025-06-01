import argparse
import torch
from networks import FullyConnected

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):

    for i, v in enumerate(inputs): # how to iterate through this tensor? that's the method I know
        print(v)

    print(inputs.size()) # get tensor size
    print(inputs.unbind()) # that doesn't work

    # Goal: prove x1_(n-1) - x1_(n-1) > 0 for all x1,x2 inputs in [0,1] x [0,1]

    print(net) # to inspect what the net looks like (for debugging only)
    print(inputs) # to inspect what the inputs look like (for debugging only)

    lower_bound = 0 # initial lower bound
    upper_bound = 1 # initial upper bound

    # recall that the inputs are normalized, so we only deal with [0, 1] x [0, 1] (I think)

    # here we have to iterate over x_1 up to x_n => what exactly is n?

    for i in range(1, n): # iterate through the inputs


        # we have to consider the following cases:
        if lower_bound <= 0: # strictly negative
            # -0.5 <= x_j <= 0
        elif lower_bound >= 0: # strictly positive
            # -0.5 <= x_j <= upper_bound * x_i
        elif lower_bound < 0 and upper_bound > 0: # the 'crossing case'
            #
        else:
            print("Bounds error.")

        # we need to incorporate it in the cases above to update the values
        # use 'forward()' from networks.py to push the inputs through the layers with the activation function
        push_input = net.forward() # I'm not sure this is the right 'forward'?
        # I want to access SPU()'s forward - is this the same in net.forward()?
        print(push_input) # for debugging

        # while exiting the loop check that x_(n-1) - x_(n) > 0 for all pairs x1, x2
        # if at least one pair breaks this relationship, return false immediately
        # otherwise true (when the computation is complete)

    '''
    return is a Boolean that is computed from:
    forall x1, x2 in the input set, it should hold that x11 > x12 <=> x11 - x12 > 0,
    i.e., x11 and x12 are the final values after the application of the transformer
    '''

    # somewhere we need: prediction - true_label < eps

    return result

def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
