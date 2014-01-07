#include "ccer.hpp"
// default constructor
namespace cv
{
	ERStat::ERStat(int init_level, int init_pixel, int init_x, int init_y) : pixel(init_pixel),
		level(init_level), area(0), perimeter(0), euler(0), probability(1.0),
		parent(0), child(0), next(0), prev(0), local_maxima(0),
		max_probability_ancestor(0), min_probability_ancestor(0),var(0)
	{
		rect = Rect(init_x,init_y,1,1);
		raw_moments[0] = 0.0;
		raw_moments[1] = 0.0;
		central_moments[0] = 0.0;
		central_moments[1] = 0.0;
		central_moments[2] = 0.0;
		crossings = new std::deque<int>();
		crossings->push_back(0);
	}
ERFilterNM::ERFilterNM()
{
	thresholdDelta = 1;
	minArea = 0.;
	maxArea = 1.;
	minProbability = 0.;
	nonMaxSuppression = false;
	minProbabilityDiff = 1.;
	num_accepted_regions = 0;
	num_rejected_regions = 0;
}

// the key method. Takes image on input, vector of ERStat is output for the first stage,
// input/output for the second one.
void ERFilterNM::run( InputArray image, vector<ERStat>& _regions )
{

	// assert correct image type
	CV_Assert( image.getMat().type() == CV_8UC1 );

	regions = &_regions;
	region_mask = Mat::zeros(image.getMat().rows+2, image.getMat().cols+2, CV_8UC1);

	// if regions vector is empty we must extract the entire component tree
	if ( regions->size() == 0 )
	{
		er_tree_extract( image );
		if (nonMaxSuppression)
		{
			vector<ERStat> aux_regions;
			regions->swap(aux_regions);
			regions->reserve(aux_regions.size());
			er_tree_nonmax_suppression( &aux_regions.front(), NULL, NULL );
			aux_regions.clear();
		}
	}
	else // if regions vector is already filled we'll just filter the current regions
	{
		// the tree root must have no parent
		CV_Assert( regions->front().parent == NULL );

		vector<ERStat> aux_regions;
		regions->swap(aux_regions);
		regions->reserve(aux_regions.size());
		er_tree_filter( image, &aux_regions.front(), NULL, NULL );
		aux_regions.clear();
	}
}

// extract the component tree and store all the ER regions
// uses the algorithm described in
// Linear time maximally stable extremal regions, D Nist¨¦r, H Stew¨¦nius ¨C ECCV 2008
void ERFilterNM::er_tree_extract( InputArray image )
{

	Mat src = image.getMat();
	// assert correct image type
	CV_Assert( src.type() == CV_8UC1 );

	if (thresholdDelta > 1)
	{
		src = (src / thresholdDelta) -1;
	}

	const unsigned char * image_data = src.data;
	int width = src.cols, height = src.rows;

	// the component stack
	vector<ERStat*> er_stack;

	//the quads for euler number calculation
	unsigned char quads[3][4];
	quads[0][0] = 1 << 3;
	quads[0][1] = 1 << 2;
	quads[0][2] = 1 << 1;
	quads[0][3] = 1;
	quads[1][0] = (1<<2)|(1<<1)|(1);
	quads[1][1] = (1<<3)|(1<<1)|(1);
	quads[1][2] = (1<<3)|(1<<2)|(1);
	quads[1][3] = (1<<3)|(1<<2)|(1<<1);
	quads[2][0] = (1<<2)|(1<<1);
	quads[2][1] = (1<<3)|(1);
	quads[2][3] = 255;


	// masks to know if a pixel is accessible and if it has been already added to some region
	vector<bool> accessible_pixel_mask(width * height);
	vector<bool> accumulated_pixel_mask(width * height);

	// heap of boundary pixels
	vector<int> boundary_pixes[256];
	vector<int> boundary_edges[256];

	// add a dummy-component before start
	er_stack.push_back(new ERStat);

	// we'll look initially for all pixels with grey-level lower than a grey-level higher than any allowed in the image
	int threshold_level = (255/thresholdDelta)+1;

	// starting from the first pixel (0,0)
	int current_pixel = 0;
	int current_edge = 0;
	int current_level = image_data[0];
	accessible_pixel_mask[0] = true;

	bool push_new_component = true;

	for (;;) {

		int x = current_pixel % width;
		int y = current_pixel / width;

		// push a component with current level in the component stack
		if (push_new_component)
			er_stack.push_back(new ERStat(current_level, current_pixel, x, y));
		push_new_component = false;

		// explore the (remaining) edges to the neighbors to the current pixel
		for (current_edge = current_edge; current_edge < 4; current_edge++)
		{

			int neighbour_pixel = current_pixel;

			switch (current_edge)
			{
			case 0: if (x < width - 1) neighbour_pixel = current_pixel + 1;  break;
			case 1: if (y < height - 1) neighbour_pixel = current_pixel + width; break;
			case 2: if (x > 0) neighbour_pixel = current_pixel - 1; break;
			default: if (y > 0) neighbour_pixel = current_pixel - width; break;
			}

			// if neighbour is not accessible, mark it accessible and retreive its grey-level value
			if ( !accessible_pixel_mask[neighbour_pixel] && (neighbour_pixel != current_pixel) )
			{

				int neighbour_level = image_data[neighbour_pixel];
				accessible_pixel_mask[neighbour_pixel] = true;

				// if neighbour level is not lower than current level add neighbour to the boundary heap
				if (neighbour_level >= current_level)
				{

					boundary_pixes[neighbour_level].push_back(neighbour_pixel);
					boundary_edges[neighbour_level].push_back(0);

					// if neighbour level is lower than our threshold_level set threshold_level to neighbour level
					if (neighbour_level < threshold_level)
						threshold_level = neighbour_level;

				}
				else // if neighbour level is lower than current add current_pixel (and next edge)
					// to the boundary heap for later processing
				{

					boundary_pixes[current_level].push_back(current_pixel);
					boundary_edges[current_level].push_back(current_edge + 1);

					// if neighbour level is lower than threshold_level set threshold_level to neighbour level
					if (current_level < threshold_level)
						threshold_level = current_level;

					// consider the new pixel and its grey-level as current pixel
					current_pixel = neighbour_pixel;
					current_edge = 0;
					current_level = neighbour_level;

					// and push a new component
					push_new_component = true;
					break;
				}
			}

		} // else neigbor was already accessible

		if (push_new_component) continue;


		// once here we can add the current pixel to the component at the top of the stack
		// but first we find how many of its neighbours are part of the region boundary (needed for
		// perimeter and crossings calc.) and the increment in quads counts for euler number calc.
		int non_boundary_neighbours = 0;
		int non_boundary_neighbours_horiz = 0;

		unsigned char quad_before[4] = {0,0,0,0};
		unsigned char quad_after[4] = {0,0,0,0};
		quad_after[0] = 1<<1;
		quad_after[1] = 1<<3;
		quad_after[2] = 1<<2;
		quad_after[3] = 1;

		for (int edge = 0; edge < 8; edge++)
		{
			int neighbour4 = -1;
			int neighbour8 = -1;
			int cell = 0;
			switch (edge)
			{
			case 0: if (x < width - 1) { neighbour4 = neighbour8 = current_pixel + 1;} cell = 5; break;
			case 1: if ((x < width - 1)&&(y < height - 1)) { neighbour8 = current_pixel + 1 + width;} cell = 8; break;
			case 2: if (y < height - 1) { neighbour4 = neighbour8 = current_pixel + width;} cell = 7; break;
			case 3: if ((x > 0)&&(y < height - 1)) { neighbour8 = current_pixel - 1 + width;} cell = 6; break;
			case 4: if (x > 0) { neighbour4 = neighbour8 = current_pixel - 1;} cell = 3; break;
			case 5: if ((x > 0)&&(y > 0)) { neighbour8 = current_pixel - 1 - width;} cell = 0; break;
			case 6: if (y > 0) { neighbour4 = neighbour8 = current_pixel - width;} cell = 1; break;
			default: if ((x < width - 1)&&(y > 0)) { neighbour8 = current_pixel + 1 - width;} cell = 2; break;
			}
			if ((neighbour4 != -1)&&(accumulated_pixel_mask[neighbour4])&&(image_data[neighbour4]<=image_data[current_pixel]))
			{
				non_boundary_neighbours++;
				if ((edge == 0) || (edge == 4))
					non_boundary_neighbours_horiz++;
			}

			int pix_value = image_data[current_pixel] + 1;
			if (neighbour8 != -1)
			{
				if (accumulated_pixel_mask[neighbour8])
					pix_value = image_data[neighbour8];
			}

			if (pix_value<=image_data[current_pixel])
			{
				switch(cell)
				{
				case 0:
					quad_before[3] = quad_before[3] | (1<<3);
					quad_after[3]  = quad_after[3]  | (1<<3);
					break;
				case 1:
					quad_before[3] = quad_before[3] | (1<<2);
					quad_after[3]  = quad_after[3]  | (1<<2);
					quad_before[0] = quad_before[0] | (1<<3);
					quad_after[0]  = quad_after[0]  | (1<<3);
					break;
				case 2:
					quad_before[0] = quad_before[0] | (1<<2);
					quad_after[0]  = quad_after[0]  | (1<<2);
					break;
				case 3:
					quad_before[3] = quad_before[3] | (1<<1);
					quad_after[3]  = quad_after[3]  | (1<<1);
					quad_before[2] = quad_before[2] | (1<<3);
					quad_after[2]  = quad_after[2]  | (1<<3);
					break;
				case 5:
					quad_before[0] = quad_before[0] | (1);
					quad_after[0]  = quad_after[0]  | (1);
					quad_before[1] = quad_before[1] | (1<<2);
					quad_after[1]  = quad_after[1]  | (1<<2);
					break;
				case 6:
					quad_before[2] = quad_before[2] | (1<<1);
					quad_after[2]  = quad_after[2]  | (1<<1);
					break;
				case 7:
					quad_before[2] = quad_before[2] | (1);
					quad_after[2]  = quad_after[2]  | (1);
					quad_before[1] = quad_before[1] | (1<<1);
					quad_after[1]  = quad_after[1]  | (1<<1);
					break;
				default:
					quad_before[1] = quad_before[1] | (1);
					quad_after[1]  = quad_after[1]  | (1);
					break;
				}
			}

		}

		int C_before[3] = {0, 0, 0};
		int C_after[3] = {0, 0, 0};

		for (int p=0; p<3; p++)
		{
			for (int q=0; q<4; q++)
			{
				if ( (quad_before[0] == quads[p][q]) && ((p<2)||(q<2)) )
					C_before[p]++;
				if ( (quad_before[1] == quads[p][q]) && ((p<2)||(q<2)) )
					C_before[p]++;
				if ( (quad_before[2] == quads[p][q]) && ((p<2)||(q<2)) )
					C_before[p]++;
				if ( (quad_before[3] == quads[p][q]) && ((p<2)||(q<2)) )
					C_before[p]++;

				if ( (quad_after[0] == quads[p][q]) && ((p<2)||(q<2)) )
					C_after[p]++;
				if ( (quad_after[1] == quads[p][q]) && ((p<2)||(q<2)) )
					C_after[p]++;
				if ( (quad_after[2] == quads[p][q]) && ((p<2)||(q<2)) )
					C_after[p]++;
				if ( (quad_after[3] == quads[p][q]) && ((p<2)||(q<2)) )
					C_after[p]++;
			}
		}

		int d_C1 = C_after[0]-C_before[0];
		int d_C2 = C_after[1]-C_before[1];
		int d_C3 = C_after[2]-C_before[2];

		er_add_pixel(er_stack.back(), x, y, non_boundary_neighbours, non_boundary_neighbours_horiz, d_C1, d_C2, d_C3);
		accumulated_pixel_mask[current_pixel] = true;

		// if we have processed all the possible threshold levels (the hea is empty) we are done!
		if (threshold_level == (255/thresholdDelta)+1)
		{

			// save the extracted regions into the output vector
			regions->reserve(num_accepted_regions+1);
			er_save(er_stack.back(), NULL, NULL);

			// clean memory
			for (size_t r=0; r<er_stack.size(); r++)
			{
				ERStat *stat = er_stack.at(r);
				if (stat->crossings)
				{
					stat->crossings->clear();
					delete(stat->crossings);
					stat->crossings = NULL;
				}
				delete stat;
			}
			er_stack.clear();

			return;
		}


		// pop the heap of boundary pixels
		current_pixel = boundary_pixes[threshold_level].back();
		boundary_pixes[threshold_level].erase(boundary_pixes[threshold_level].end()-1);
		current_edge  = boundary_edges[threshold_level].back();
		boundary_edges[threshold_level].erase(boundary_edges[threshold_level].end()-1);

		while (boundary_pixes[threshold_level].empty() && (threshold_level < (255/thresholdDelta)+1))
			threshold_level++;


		int new_level = image_data[current_pixel];

		// if the new pixel has higher grey value than the current one
		if (new_level != current_level) {

			current_level = new_level;

			// process components on the top of the stack until we reach the higher grey-level
			while (er_stack.back()->level < new_level)
			{
				ERStat* er = er_stack.back();
				er_stack.erase(er_stack.end()-1);

				if (new_level < er_stack.back()->level)
				{
					er_stack.push_back(new ERStat(new_level, current_pixel, current_pixel%width, current_pixel/width));
					er_merge(er_stack.back(), er);
					break;
				}

				er_merge(er_stack.back(), er);
			}

		}

	}
}

// accumulate a pixel into an ER
void ERFilterNM::er_add_pixel(ERStat *parent, int x, int y, int non_border_neighbours,
							  int non_border_neighbours_horiz,
							  int d_C1, int d_C2, int d_C3)
{
	parent->area++;
	parent->perimeter += 4 - 2*non_border_neighbours;

	if (parent->crossings->size()>0)
	{
		if (y<parent->rect.y) parent->crossings->push_front(2);
		else if (y>parent->rect.br().y-1) parent->crossings->push_back(2);
		else {
			parent->crossings->at(y - parent->rect.y) += 2-2*non_border_neighbours_horiz;
		}
	} else {
		parent->crossings->push_back(2);
	}

	parent->euler += (d_C1 - d_C2 + 2*d_C3) / 4;

	int new_x1 = min(parent->rect.x,x);
	int new_y1 = min(parent->rect.y,y);
	int new_x2 = max(parent->rect.br().x-1,x);
	int new_y2 = max(parent->rect.br().y-1,y);
	parent->rect.x = new_x1;
	parent->rect.y = new_y1;
	parent->rect.width  = new_x2-new_x1+1;
	parent->rect.height = new_y2-new_y1+1;

	parent->raw_moments[0] += x;
	parent->raw_moments[1] += y;

	parent->central_moments[0] += x * x;
	parent->central_moments[1] += x * y;
	parent->central_moments[2] += y * y;
}

// merge an ER with its nested parent
void ERFilterNM::er_merge(ERStat *parent, ERStat *child)
{

	parent->area += child->area;

	parent->perimeter += child->perimeter;


	for (int i=parent->rect.y; i<=min(parent->rect.br().y-1,child->rect.br().y-1); i++)
		if (i-child->rect.y >= 0)
			parent->crossings->at(i-parent->rect.y) += child->crossings->at(i-child->rect.y);

	for (int i=parent->rect.y-1; i>=child->rect.y; i--)
		if (i-child->rect.y < (int)child->crossings->size())
			parent->crossings->push_front(child->crossings->at(i-child->rect.y));
		else
			parent->crossings->push_front(0);

	for (int i=parent->rect.br().y; i<child->rect.y; i++)
		parent->crossings->push_back(0);

	for (int i=max(parent->rect.br().y,child->rect.y); i<=child->rect.br().y-1; i++)
		parent->crossings->push_back(child->crossings->at(i-child->rect.y));

	parent->euler += child->euler;

	int new_x1 = min(parent->rect.x,child->rect.x);
	int new_y1 = min(parent->rect.y,child->rect.y);
	int new_x2 = max(parent->rect.br().x-1,child->rect.br().x-1);
	int new_y2 = max(parent->rect.br().y-1,child->rect.br().y-1);
	parent->rect.x = new_x1;
	parent->rect.y = new_y1;
	parent->rect.width  = new_x2-new_x1+1;
	parent->rect.height = new_y2-new_y1+1;

	parent->raw_moments[0] += child->raw_moments[0];
	parent->raw_moments[1] += child->raw_moments[1];

	parent->central_moments[0] += child->central_moments[0];
	parent->central_moments[1] += child->central_moments[1];
	parent->central_moments[2] += child->central_moments[2];

	vector<int> m_crossings;
	m_crossings.push_back(child->crossings->at((int)(child->rect.height)/6));
	m_crossings.push_back(child->crossings->at((int)3*(child->rect.height)/6));
	m_crossings.push_back(child->crossings->at((int)5*(child->rect.height)/6));
	sort(m_crossings.begin(), m_crossings.end());
	child->med_crossings = (float)m_crossings.at(1);

	// free unnecessary mem
	child->crossings->clear();
	delete(child->crossings);
	child->crossings = NULL;

	// recover the original grey-level
	child->level = child->level*thresholdDelta;

	// before saving calculate P(child|character) and filter if possible
	if (classifier != NULL)
	{
		child->probability = classifier->eval(*child);
	}

	if ( (((classifier!=NULL)?(child->probability >= minProbability):true)||(nonMaxSuppression)) &&
		((child->area >= (minArea*region_mask.rows*region_mask.cols)) &&
		(child->area <= (maxArea*region_mask.rows*region_mask.cols)) &&
		(child->rect.width > 2) && (child->rect.height > 2)) )
	{

		num_accepted_regions++;

		child->next = parent->child;
		if (parent->child)
			parent->child->prev = child;
		parent->child = child;
		child->parent = parent;

	} else {

		num_rejected_regions++;

		if (child->prev !=NULL)
			child->prev->next = child->next;

		ERStat *new_child = child->child;
		if (new_child != NULL)
		{
			while (new_child->next != NULL)
				new_child = new_child->next;
			new_child->next = parent->child;
			if (parent->child)
				parent->child->prev = new_child;
			parent->child   = child->child;
			child->child->parent = parent;
		}

		// free mem
		if(child->crossings)
		{
			child->crossings->clear();
			delete(child->crossings);
			child->crossings = NULL;
		}
		delete(child);
	}

}

// copy extracted regions into the output vector
ERStat* ERFilterNM::er_save( ERStat *er, ERStat *parent, ERStat *prev )
{

	regions->push_back(*er);

	regions->back().parent = parent;
	if (prev != NULL)
	{
		prev->next = &(regions->back());
	}
	else if (parent != NULL)
		parent->child = &(regions->back());

	ERStat *old_prev = NULL;
	ERStat *this_er  = &regions->back();

	if (this_er->parent == NULL)
	{
		this_er->probability = 0;
	}

	if (nonMaxSuppression)
	{
		if (this_er->parent == NULL)
		{
			this_er->max_probability_ancestor = this_er;
			this_er->min_probability_ancestor = this_er;
		}
		else
		{
			this_er->max_probability_ancestor = (this_er->probability > parent->max_probability_ancestor->probability)? this_er :  parent->max_probability_ancestor;

			this_er->min_probability_ancestor = (this_er->probability < parent->min_probability_ancestor->probability)? this_er :  parent->min_probability_ancestor;

			if ( (this_er->max_probability_ancestor->probability > minProbability) && (this_er->max_probability_ancestor->probability - this_er->min_probability_ancestor->probability > minProbabilityDiff))
			{
				this_er->max_probability_ancestor->local_maxima = true;
				if ((this_er->max_probability_ancestor == this_er) && (this_er->parent->local_maxima))
				{
					this_er->parent->local_maxima = false;
				}
			}
			else if (this_er->probability < this_er->parent->probability)
			{
				this_er->min_probability_ancestor = this_er;
			}
			else if (this_er->probability > this_er->parent->probability)
			{
				this_er->max_probability_ancestor = this_er;
			}


		}
	}

	for (ERStat * child = er->child; child; child = child->next)
	{
		old_prev = er_save(child, this_er, old_prev);
	}

	return this_er;
}

// recursively walk the tree and filter (remove) regions using the callback classifier
ERStat* ERFilterNM::er_tree_filter ( InputArray image, ERStat * stat, ERStat *parent, ERStat *prev )
{
	Mat src = image.getMat();
	// assert correct image type
	CV_Assert( src.type() == CV_8UC1 );

	//Fill the region and calculate 2nd stage features
	Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x+2,stat->rect.br().y+2)));
	region = Scalar(0);
	int newMaskVal = 255;
	int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
	Rect rect;

	floodFill( src(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))),
		region, Point(stat->pixel%src.cols - stat->rect.x, stat->pixel/src.cols - stat->rect.y),
		Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );
	rect.width += 2;
	rect.height += 2;
	region = region(rect);

	vector<vector<Point> > contours;
	vector<Point> contour_poly;
	vector<Vec4i> hierarchy;
	findContours( region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );
	//TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission
	//     if the region is very small because otherwise we'll loose all the convexities
	approxPolyDP( Mat(contours[0]), contour_poly, (float)min(rect.width,rect.height)/17, true );

	bool was_convex = false;
	int  num_inflexion_points = 0;

	for (int p = 0 ; p<(int)contour_poly.size(); p++)
	{
		int p_prev = p-1;
		int p_next = p+1;
		if (p_prev == -1)
			p_prev = (int)contour_poly.size()-1;
		if (p_next == (int)contour_poly.size())
			p_next = 0;

		double angle_next = atan2((double)(contour_poly[p_next].y-contour_poly[p].y),
			(double)(contour_poly[p_next].x-contour_poly[p].x));
		double angle_prev = atan2((double)(contour_poly[p_prev].y-contour_poly[p].y),
			(double)(contour_poly[p_prev].x-contour_poly[p].x));
		if ( angle_next < 0 )
			angle_next = 2.*CV_PI + angle_next;

		double angle = (angle_next - angle_prev);
		if (angle > 2.*CV_PI)
			angle = angle - 2.*CV_PI;
		else if (angle < 0)
			angle = 2.*CV_PI + std::abs(angle);

		if (p>0)
		{
			if ( ((angle > CV_PI)&&(!was_convex)) || ((angle < CV_PI)&&(was_convex)) )
				num_inflexion_points++;
		}
		was_convex = (angle > CV_PI);

	}

	floodFill(region, Point(0,0), Scalar(255), 0);
	int holes_area = region.cols*region.rows-countNonZero(region);

	int hull_area = 0;

	{

		vector<Point> hull;
		convexHull(contours[0], hull, false);
		hull_area = (int)contourArea(hull);
	}


	stat->hole_area_ratio = (float)holes_area / stat->area;
	stat->convex_hull_ratio = (float)hull_area / (float)contourArea(contours[0]);
	stat->num_inflexion_points = (float)num_inflexion_points;


	// calculate P(child|character) and filter if possible
	if ( (classifier != NULL) && (stat->parent != NULL) )
	{
		stat->probability = classifier->eval(*stat);
	}

	if ( ( ((classifier != NULL)?(stat->probability >= minProbability):true) &&
		((stat->area >= minArea*region_mask.rows*region_mask.cols) &&
		(stat->area <= maxArea*region_mask.rows*region_mask.cols)) ) ||
		(stat->parent == NULL) )
	{

		num_accepted_regions++;
		regions->push_back(*stat);

		regions->back().parent = parent;
		regions->back().next   = NULL;
		regions->back().child  = NULL;

		if (prev != NULL)
			prev->next = &(regions->back());
		else if (parent != NULL)
			parent->child = &(regions->back());

		ERStat *old_prev = NULL;
		ERStat *this_er  = &regions->back();

		for (ERStat * child = stat->child; child; child = child->next)
		{
			old_prev = er_tree_filter(image, child, this_er, old_prev);
		}

		return this_er;

	} else {

		num_rejected_regions++;

		ERStat *old_prev = prev;

		for (ERStat * child = stat->child; child; child = child->next)
		{
			old_prev = er_tree_filter(image, child, parent, old_prev);
		}

		return old_prev;
	}

}

// recursively walk the tree selecting only regions with local maxima probability
ERStat* ERFilterNM::er_tree_nonmax_suppression ( ERStat * stat, ERStat *parent, ERStat *prev )
{

	if ( ( stat->local_maxima ) || ( stat->parent == NULL ) )
	{

		regions->push_back(*stat);

		regions->back().parent = parent;
		regions->back().next   = NULL;
		regions->back().child  = NULL;

		if (prev != NULL)
			prev->next = &(regions->back());
		else if (parent != NULL)
			parent->child = &(regions->back());

		ERStat *old_prev = NULL;
		ERStat *this_er  = &regions->back();

		for (ERStat * child = stat->child; child; child = child->next)
		{
			old_prev = er_tree_nonmax_suppression( child, this_er, old_prev );
		}

		return this_er;

	} else {

		num_rejected_regions++;
		num_accepted_regions--;

		ERStat *old_prev = prev;

		for (ERStat * child = stat->child; child; child = child->next)
		{
			old_prev = er_tree_nonmax_suppression( child, parent, old_prev );
		}

		return old_prev;
	}

}

void ERFilterNM::setCallback(const Ptr<ERFilter::Callback>& cb)
{
	classifier = cb;
};

void ERFilterNM::setMinArea(float _minArea)
{
	CV_Assert( (_minArea >= 0) && (_minArea < maxArea) );
	minArea = _minArea;
	return;
};

void ERFilterNM::setMaxArea(float _maxArea)
{
	CV_Assert(_maxArea <= 1);
	CV_Assert(minArea < _maxArea);
	maxArea = _maxArea;
	return;
};

void ERFilterNM::setThresholdDelta(int _thresholdDelta)
{
	CV_Assert( (_thresholdDelta > 0) && (_thresholdDelta <= 128) );
	thresholdDelta = _thresholdDelta;
	return;
};

void ERFilterNM::setMinProbability(float _minProbability)
{
	CV_Assert( (_minProbability >= 0.0) && (_minProbability <= 1.0) );
	minProbability = _minProbability;
	return;
};

void ERFilterNM::setMinProbabilityDiff(float _minProbabilityDiff)
{
	CV_Assert( (_minProbabilityDiff >= 0.0) && (_minProbabilityDiff <= 1.0) );
	minProbabilityDiff = _minProbabilityDiff;
	return;
};

void ERFilterNM::setNonMaxSuppression(bool _nonMaxSuppression)
{
	nonMaxSuppression = _nonMaxSuppression;
	return;
};

int ERFilterNM::getNumRejected()
{
	return num_rejected_regions;
};


void ERFilterNM::FilterRegion(vector<ERStat>& _regions){
	regions = &_regions;
	CV_Assert( regions->front().parent == NULL );
	vector<ERStat> aux_regions;
	regions->swap(aux_regions);
	regions->reserve(aux_regions.size());
	FilterRegionHelper(&aux_regions.front(), NULL, NULL);
	aux_regions.clear();
}
ERStat* ERFilterNM::FilterRegionHelper( ERStat * stat, ERStat *parent, ERStat *prev){
	if (((stat->area >= minArea*region_mask.rows*region_mask.cols) 	&&(stat->area <= maxArea*region_mask.rows*region_mask.cols))
		||(stat->parent == NULL)){
		num_accepted_regions++;
		regions->push_back(*stat);

		regions->back().parent = parent;
		regions->back().next   = NULL;
		regions->back().child  = NULL;

		if (prev != NULL)
			prev->next = &(regions->back());
		else if (parent != NULL)
			parent->child = &(regions->back());

		ERStat *old_prev = NULL;
		ERStat *this_er  = &regions->back();

		for (ERStat * child = stat->child; child; child = child->next){
			old_prev = FilterRegionHelper(child, this_er, old_prev);
		}
		return this_er;

	} else {
		num_rejected_regions++;
		ERStat *old_prev = prev;

		for (ERStat * child = stat->child; child; child = child->next){
			old_prev = FilterRegionHelper(child, parent, old_prev);
		}
		return old_prev;
	}
}


}

