package hu.sztaki.ilab.ps.matrix.factorization.utils

import hu.sztaki.ilab.ps.matrix.factorization.utils.Utils.{ItemId, UserId}


/**
  * Rating type for training data
  */
case class Rating(user: UserId, item: ItemId, rating: Double, label: String, timestamp: Long) {
  def enrich(workerId: Int, ratingId: Double) = RichRating(user, item, rating, label, workerId, ratingId, timestamp)
}

/**
  * A rating with a target worker and a rating ID
  */
case class RichRating(user: UserId, item: ItemId, rating: Double, label: String, targetWorker: Int, ratingId: Double,
                      timestamp: Long) {
  def reduce() = Rating(user, item, rating, label, timestamp)
}

object Rating {

  /**
    * Constructs a rating from a tuple
    *
    * @param t
    * A tuple containing a user ID, an item ID, and a rating
    * @return
    * A Rating with the same values and a timestamp of 0
    */
  def fromTuple(t: (Int,Int,Double,String)): Rating = Rating(t._1, t._2, t._3, t._4, 0)
}